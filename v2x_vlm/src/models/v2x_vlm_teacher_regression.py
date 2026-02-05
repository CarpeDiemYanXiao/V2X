# src/models/v2x_vlm_teacher_regression.py
"""
直接回归坐标的Teacher模型
不使用分类，直接预测坐标值
可能比分类方法更简单有效
"""
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM

class V2XVLMTeacherRegression(nn.Module):
    """
    Teacher V2X-VLM (回归式架构)
    Florence-2-large + 直接回归坐标
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.device)
        self.model_path = cfg.teacher_model_path
        self.future_steps = cfg.future_steps

        print(f"Loading Teacher Florence-2 (Regression) from {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(self.device)

        hidden_dim = self.model.language_model.config.hidden_size
        
        # ========== 特征融合层（简化，减少过拟合） ==========
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # ========== 直接回归坐标的预测头（更深的网络，更强的表达能力） ==========
        # 输出维度：future_steps * 2 (x, y坐标)
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, self.future_steps * 2)  # 直接输出坐标
        )
        
        # 初始化：更激进的初始化策略，让所有层都有更大的权重
        for module in self.traj_head.modules():
            if isinstance(module, nn.Linear):
                if module == self.traj_head[-1]:  # 最后一层
                    # 使用更大的std，让初始输出范围在[-50, 50]左右
                    nn.init.normal_(module.weight, mean=0.0, std=0.5)
                    nn.init.constant_(module.bias, 0.0)
                else:
                    # 中间层也使用更大的初始化
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
        # 特征融合层也使用更大的初始化
        for module in self.feature_fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, images, prompts):
        # 处理输入
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        batch_size = inputs["pixel_values"].shape[0]
        decoder_start_token_id = self.model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.model.config.bos_token_id
        
        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=self.device
        )

        # 前向传播
        outputs = self.model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )

        # 获取hidden states
        encoder_hidden = None
        decoder_hidden = None
        
        if hasattr(outputs, 'encoder_last_hidden_state') and outputs.encoder_last_hidden_state is not None:
            encoder_hidden = outputs.encoder_last_hidden_state
        
        if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            decoder_hidden = outputs.last_hidden_state
        elif hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
            decoder_hidden = outputs.decoder_hidden_states[-1]
        
        if decoder_hidden is None:
            raise ValueError("无法从教师模型输出中获取隐藏状态")
        
        # 简化特征提取：直接使用decoder的最后一个token
        if decoder_hidden.dim() == 3:
            decoder_pooled = decoder_hidden[:, -1, :]  # 只用最后一个token
        else:
            decoder_pooled = decoder_hidden
        
        # 如果有encoder hidden，简单拼接而不是加权融合
        if encoder_hidden is not None and encoder_hidden.dim() == 3:
            encoder_pooled = encoder_hidden.mean(dim=1)  # 简单平均
            # 拼接两个特征
            fused = decoder_pooled + encoder_pooled  # 残差连接
        else:
            fused = decoder_pooled
        
        pooled = self.feature_fusion(fused)

        # 直接回归坐标
        flat_coords = self.traj_head(pooled)  # [B, future_steps * 2]
        coords = flat_coords.view(-1, self.future_steps, 2)  # [B, future_steps, 2]

        return {
            "trajectory_coords": coords  # [B, 45, 2]
        }
