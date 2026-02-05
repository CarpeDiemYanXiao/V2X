# src/models/v2x_vlm_teacher.py
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM

class V2XVLMTeacher(nn.Module):
    """
    Teacher V2X-VLM (生成式架构)
    Florence-2-large + Token Classification Head
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.device)
        self.model_path = cfg.teacher_model_path
        self.future_steps = cfg.future_steps
        # [Step 4 新增]
        self.vocab_size = cfg.model.vocab_size

        print(f"Loading Teacher Florence-2 from {self.model_path}")

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
        
        # ========== 改进的特征融合层 ==========
        # 使用更强大的特征融合，结合视觉和文本信息
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # ========== 轨迹预测头 (改进版) ==========
        # 增加模型容量，使用更深的网络
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),  # 最后一层用更小的dropout
            nn.Linear(hidden_dim, self.future_steps * 2 * self.vocab_size)
        )
        
        # 初始化：使用更好的初始化策略
        for module in self.traj_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # 最后一层特殊初始化：使用更小的初始值避免输出过大
        nn.init.normal_(self.traj_head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.traj_head[-1].bias, 0.0)

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

        # 尝试获取encoder和decoder的hidden states
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
        
        # 改进的特征提取策略：
        # 1. 使用decoder的最后一个token（包含生成信息）
        # 2. 如果可用，结合encoder的输出（包含视觉-文本融合信息）
        if decoder_hidden.dim() == 3:
            # 使用decoder最后一个token
            decoder_pooled = decoder_hidden[:, -1, :]  # [B, hidden_dim]
        else:
            decoder_pooled = decoder_hidden
        
        # 如果encoder输出可用，进行融合
        if encoder_hidden is not None and encoder_hidden.dim() == 3:
            # 使用encoder输出的mean pooling（包含视觉-文本融合信息）
            encoder_pooled = encoder_hidden.mean(dim=1)  # [B, hidden_dim]
            # 融合encoder和decoder特征
            fused = (decoder_pooled + encoder_pooled) / 2.0
        else:
            fused = decoder_pooled
        
        # 通过特征融合层进一步处理
        pooled = self.feature_fusion(fused)

        # ========== 生成 logits ==========
        flat_logits = self.traj_head(pooled)
        logits = flat_logits.view(-1, self.future_steps * 2, self.vocab_size)

        return {
            "trajectory_logits": logits # [B, 90, 1024]
        }