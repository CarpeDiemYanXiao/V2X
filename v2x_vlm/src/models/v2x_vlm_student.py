# src/models/v2x_vlm_student.py
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM

class V2XVLMStudent(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.device)
        self.future_steps = cfg.future_steps
        self.proj_dim = cfg.model.proj_dim 
        
        # [Step 4 新增] 获取词表大小，用于分类头
        self.vocab_size = cfg.model.vocab_size

        print(f"[Student] Loading Florence-2 from {cfg.student_model_path}")

        self.processor = AutoProcessor.from_pretrained(
            cfg.student_model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.student_model_path,
            trust_remote_code=True
        ).to(self.device)

        hidden_dim = self.model.config.text_config.d_model if hasattr(self.model.config, 'text_config') else 768

        # ========== 1. 对比学习投影头 (保持 Step 3 的修复) ==========
        with torch.no_grad():
            # 动态获取 Vision Tower 输出维度
            dummy_img = torch.zeros(1, 3, 768, 768).to(self.device)
            vision_dim = hidden_dim # 默认 fallback
            try:
                # 尝试推断
                vision_tower = self.model.model.vision_tower
                if hasattr(vision_tower, "config"):
                    vision_dim = vision_tower.config.hidden_size
            except:
                pass
            
        self.visual_projection = nn.Linear(vision_dim, self.proj_dim)
        self.text_projection = nn.Linear(hidden_dim, self.proj_dim)

        # ========== 2. 轨迹预测头 (Step 4 核心修改：生成式分类头) ==========
        # 输出维度：序列长度(steps*2) * 词表大小(vocab_size)
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            # [Step 4] 改为分类输出：预测每个坐标点属于哪个 bin 的概率
            nn.Linear(hidden_dim, self.future_steps * 2 * self.vocab_size)
        )

        self._freeze_vision_encoder()

    def _freeze_vision_encoder(self):
        for name, param in self.model.named_parameters():
            if "vision" in name.lower() or "davit" in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True

        for p in self.visual_projection.parameters():
            p.requires_grad = True
        for p in self.text_projection.parameters():
            p.requires_grad = True
        for p in self.traj_head.parameters():
            p.requires_grad = True

    def forward(self, images, prompts):
        # 1. 预处理
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
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

        # ========== 独立特征提取 (用于对比学习) ==========
        # A. 视觉特征
        vision_tower = self.model.model.vision_tower
        vision_outputs = vision_tower(inputs["pixel_values"])
        if isinstance(vision_outputs, tuple):
            vision_features = vision_outputs[0]
        else:
            vision_features = vision_outputs
        
        z_visual = vision_features.mean(dim=1)
        z_visual_proj = self.visual_projection(z_visual)

        # B. 文本特征
        text_embedder = self.model.get_input_embeddings()
        text_features = text_embedder(inputs["input_ids"])
        mask = inputs["attention_mask"].unsqueeze(-1).expand(text_features.size()).float()
        z_text = (text_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        z_text_proj = self.text_projection(z_text)

        # ========== 融合特征提取 (用于轨迹生成) ==========
        outputs = self.model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )

        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, 'decoder_hidden_states'):
            hidden = outputs.decoder_hidden_states[-1]
        else:
            hidden = outputs[0]

        pooled_fused = hidden.mean(dim=1)

        # ========== 预测 logits (Step 4 修改) ==========
        # 1. 获得扁平的 logits: [B, steps * 2 * vocab_size]
        flat_logits = self.traj_head(pooled_fused)
        
        # 2. Reshape 为分类所需的形状: [B, sequence_len, num_classes]
        # sequence_len = future_steps * 2 (即 x0, y0, x1, y1...)
        logits = flat_logits.view(-1, self.future_steps * 2, self.vocab_size)

        # 3. 获取预测的 Token ID (用于推理/可视化，不用于训练 Loss)
        pred_tokens = torch.argmax(logits, dim=-1)

        return {
            "trajectory_logits": logits,      # [B, 90, 1024] 用于 CrossEntropy / KD
            "pred_tokens": pred_tokens,       # [B, 90]       用于可视化
            "visual_features": z_visual_proj, # [B, 256]      用于对比损失
            "text_features": z_text_proj,     # [B, 256]      用于对比损失
            "alignment": pooled_fused,
        }