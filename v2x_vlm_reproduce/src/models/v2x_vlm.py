"""
V2X-VLM 主模型

论文 Section 4 METHOD:
整合 Florence-2 backbone, 轨迹解码器, 特征对齐模块

架构概览:
1. 图像拼接: [I_v, I_i] ∈ R^{H × (W_v + W_i) × 3}
2. Florence-2 编码: 
   - DaViT视觉编码器 f_v(·)
   - BERT文本编码器 f_t(·)
   - 多模态Transformer融合
3. 轨迹解码: f_traj(F_multi) → τ
4. 知识蒸馏: Teacher (Florence-2-Large, frozen) → Student (Florence-2-Base, trainable)

支持设备: CUDA / NPU (华为昇腾) / CPU
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Dict, Optional, Tuple, List
import copy

# NPU 兼容性导入
try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

from .trajectory_head import TrajectoryHead
from .feature_alignment import FeatureAlignment


def get_device_type(device: str) -> str:
    """获取设备类型 (不含设备编号)"""
    if 'npu' in str(device):
        return 'npu'
    elif 'cuda' in str(device):
        return 'cuda'
    return 'cpu'


class V2XVLM(nn.Module):
    """
    V2X-VLM: End-to-End V2X Cooperative Autonomous Driving through Large Vision-Language Models
    
    论文核心创新:
    1. 统一的VLM处理V2X多视图输入
    2. Teacher-Student知识蒸馏框架
    3. 对比特征对齐
    4. MLP轨迹解码器
    
    支持设备: CUDA / NPU / CPU
    
    Attributes:
        student_model: Florence-2-Base (可训练)
        teacher_model: Florence-2-Large (冻结)
        trajectory_head: 轨迹解码器
        alignment_module: 特征对齐模块
    """
    
    def __init__(
        self,
        student_model_name: str = "microsoft/Florence-2-base",
        teacher_model_name: str = "microsoft/Florence-2-large",
        trajectory_length: int = 45,
        hidden_dim: int = 768,           # Florence-2-base hidden dim
        teacher_hidden_dim: int = 1024,  # Florence-2-large hidden dim
        projection_dim: int = 256,
        temperature: float = 0.07,       # κ for contrastive loss
        kd_temperature: float = 2.0,     # T for knowledge distillation
        freeze_teacher: bool = True,
        use_knowledge_distillation: bool = True,
        use_contrastive_alignment: bool = True,
        device: str = "cuda",
        cache_dir: str = None            # 模型缓存目录
    ):
        super().__init__()
        
        self.trajectory_length = trajectory_length
        self.hidden_dim = hidden_dim
        self.teacher_hidden_dim = teacher_hidden_dim
        self.kd_temperature = kd_temperature
        self.use_kd = use_knowledge_distillation
        self.use_alignment = use_contrastive_alignment
        self.device = device
        self.device_type = get_device_type(device)
        
        # NPU 特定设置
        if self.device_type == 'npu' and NPU_AVAILABLE:
            print("NPU mode enabled")
        
        # 设置模型缓存目录 (默认为项目目录下的 pretrained_models)
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "pretrained_models"
            )
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        print(f"Model cache directory: {cache_dir}")
        
        print("Loading Student model (Florence-2-Base)...")
        # Student Model - Florence-2-Base (trainable)
        # 使用 attn_implementation="eager" 避免 SDPA 兼容性问题
        self.student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            attn_implementation="eager",
            local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(
            student_model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=True
        )
                
        # Teacher Model - Florence-2-Large (frozen)
        if use_knowledge_distillation:
            print("Loading Teacher model (Florence-2-Large)...")
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
                attn_implementation="eager",
                local_files_only=True
            )
            if freeze_teacher:
                for param in self.teacher_model.parameters():
                    param.requires_grad = False
                self.teacher_model.eval()
        else:
            self.teacher_model = None
        
        # 轨迹解码头
        # 论文 Section 4.2: "simple Trajectory Decoder f_traj(·) based on MLP"
        self.trajectory_head = TrajectoryHead(
            hidden_dim=hidden_dim,
            trajectory_length=trajectory_length,
            mlp_hidden_dims=(512, 256, 128),
            dropout=0.1
        )
        
        # 特征对齐模块 (用于对比学习)
        # 论文 Section 4.3: 对齐 student 视觉编码器 (768d) 与 teacher 文本表示 (1024d)
        if use_contrastive_alignment:
            self.alignment_module = FeatureAlignment(
                vision_dim=hidden_dim,               # student: 768
                text_dim=teacher_hidden_dim,          # teacher: 1024
                projection_dim=projection_dim,
                temperature=temperature
            )
        else:
            self.alignment_module = None
        
        # 知识蒸馏维度映射 (如果teacher和student维度不同)
        if use_knowledge_distillation and teacher_hidden_dim != hidden_dim:
            self.kd_proj = nn.Linear(hidden_dim, teacher_hidden_dim)
        else:
            self.kd_proj = None
    
    def get_processor(self):
        """获取Florence-2 processor"""
        return self.processor
    
    def encode_multimodal(
        self,
        model: nn.Module,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        分步提取三种独立的多模态特征

        论文 Section 4.2:
        - 视觉编码: F_v = f_v([I_v, I_i])           → vision_features  (纯视觉)
        - 文本编码: F_t = f_t(E)                     → text_embeddings  (纯文本)
        - 多模态融合: F_multi = Decoder(Enc(F_v,F_t)) → fusion_features  (解码器输出)

        Florence-2 内部结构:
        - vision_tower (DaViT): 图像编码
        - image_proj_norm + image_projection: 投影到语言模型维度
        - language_model (BART-like): encoder 处理 [image;text], decoder 生成
        """
        model_hidden_dim = self.hidden_dim if model == self.student_model else self.teacher_hidden_dim
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        dtype = pixel_values.dtype

        # ========== Step 1: 纯视觉特征 F_v (DaViT + projection) ==========
        try:
            if hasattr(model, '_encode_image'):
                # Florence-2 自带方法: vision_tower → reshape → norm → projection
                vision_features = model._encode_image(pixel_values)
            else:
                vision_features = model.vision_tower(pixel_values)
                if isinstance(vision_features, tuple):
                    vision_features = vision_features[0]
                if hasattr(vision_features, 'last_hidden_state'):
                    vision_features = vision_features.last_hidden_state
                if hasattr(model, 'image_proj_norm'):
                    vision_features = model.image_proj_norm(vision_features)
                if hasattr(model, 'image_projection'):
                    vision_features = model.image_projection(vision_features)
        except Exception as e:
            print(f"Warning: Vision feature extraction failed: {e}")
            vision_features = torch.zeros(batch_size, 1, model_hidden_dim, device=device, dtype=dtype)

        # 确保 [B, N_v, D]
        if vision_features.dim() == 4:
            B, C, H, W = vision_features.shape
            vision_features = vision_features.flatten(2).permute(0, 2, 1)
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(1)

        # ========== Step 2: 纯文本嵌入 F_t (embed_tokens, 无上下文混合) ==========
        try:
            encoder = model.language_model.model.encoder
            text_embeddings = encoder.embed_tokens(input_ids)  # [B, N_t, D]
        except Exception as e:
            print(f"Warning: Text embedding extraction failed: {e}")
            text_embeddings = torch.zeros(batch_size, input_ids.shape[1], model_hidden_dim, device=device, dtype=dtype)

        # ========== Step 3: 融合特征 F_multi (encoder-decoder pipeline) ==========
        try:
            # 3a. 拼接 [image_tokens; text_tokens] 送入 encoder
            encoder_inputs = torch.cat([vision_features, text_embeddings], dim=1)
            image_mask = torch.ones(
                batch_size, vision_features.shape[1], device=device, dtype=attention_mask.dtype
            )
            combined_mask = torch.cat([image_mask, attention_mask], dim=1)

            encoder_outputs = encoder(
                inputs_embeds=encoder_inputs,
                attention_mask=combined_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # 3b. Decoder: 用 input_ids 的 embedding 作为 decoder 输入
            decoder = model.language_model.model.decoder
            decoder_input_embeds = encoder.embed_tokens(input_ids)
            decoder_outputs = decoder(
                inputs_embeds=decoder_input_embeds,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=combined_mask,
                output_hidden_states=True,
                return_dict=True
            )
            fusion_features = decoder_outputs.last_hidden_state  # [B, N_dec, D]
        except Exception as e:
            print(f"Warning: Fusion extraction via manual pipeline failed: {e}")
            # Fallback: 调用完整 forward
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    decoder_input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True
                )
                if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
                    fusion_features = outputs.decoder_hidden_states[-1]
                else:
                    fusion_features = torch.cat([vision_features, text_embeddings], dim=1)
            except Exception as e2:
                print(f"Warning: Full forward also failed: {e2}")
                fusion_features = torch.cat([vision_features, text_embeddings], dim=1)

        if fusion_features.dim() == 2:
            fusion_features = fusion_features.unsqueeze(1)

        return {
            'vision_features': vision_features,   # [B, N_v, D] 纯视觉
            'text_embeddings': text_embeddings,    # [B, N_t, D] 纯文本
            'fusion_features': fusion_features     # [B, N_d, D] 多模态融合
        }
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trajectory_gt: Optional[torch.Tensor] = None,
        return_features: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 — MLP 回归轨迹预测

        Args:
            pixel_values: 拼接的图像 [B, 3, H, 2W]
            input_ids: 文本 token IDs [B, L]
            attention_mask: 注意力掩码 [B, L]
            trajectory_gt: GT 轨迹 [B, T, 2]
            return_features: 是否返回中间特征
        """
        outputs = {}
        losses = {}

        # ========== 1. Student 多模态编码 + MLP 轨迹预测 ==========
        student_features = self.encode_multimodal(
            self.student_model, pixel_values, input_ids, attention_mask
        )
        trajectory_pred = self.trajectory_head(
            student_features['fusion_features'], attention_mask=None
        )
        outputs['trajectory_pred'] = trajectory_pred

        if trajectory_gt is not None:
            # 时间步加权: 远端时间步权重更高, 提升远距离预测精度
            T = trajectory_pred.shape[1]  # 45
            time_weights = 1.0 + 0.5 * torch.linspace(0, 1, T, device=trajectory_pred.device)
            time_weights = time_weights.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            
            # 加权 smooth_l1 loss
            per_step_loss = F.smooth_l1_loss(trajectory_pred, trajectory_gt, reduction='none')
            losses['loss_traj'] = (per_step_loss * time_weights).mean()
            
            # 速度一致性正则: 鼓励轨迹平滑
            pred_vel = trajectory_pred[:, 1:, :] - trajectory_pred[:, :-1, :]
            gt_vel = trajectory_gt[:, 1:, :] - trajectory_gt[:, :-1, :]
            losses['loss_vel'] = F.smooth_l1_loss(pred_vel, gt_vel) * 0.5

        # ========== 2. Teacher Forward (对比对齐 + KD) ==========
        need_teacher = (self.use_kd or self.use_alignment) and self.teacher_model is not None
        teacher_features = None

        if need_teacher:
            with torch.no_grad():
                teacher_features = self.encode_multimodal(
                    self.teacher_model, pixel_values, input_ids, attention_mask
                )

        # ========== 3. 对比对齐损失 L_align: student vision vs teacher text ==========
        if (self.use_alignment and self.alignment_module is not None
                and teacher_features is not None):
            z_v, z_t = self.alignment_module(
                student_features['vision_features'],
                teacher_features['text_embeddings']
            )
            losses['loss_align'] = self.alignment_module.compute_loss(z_v, z_t)

        # ========== 4. 知识蒸馏损失 L_KD: 隐层特征 KL ==========
        if self.use_kd and teacher_features is not None:
            losses['loss_kd'] = self.compute_kd_loss(
                student_features['fusion_features'],
                teacher_features['fusion_features']
            )

        outputs['losses'] = losses

        if return_features:
            outputs['student_features'] = student_features
            if teacher_features is not None:
                outputs['teacher_features'] = teacher_features

        return outputs
    
    def compute_kd_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算知识蒸馏损失
        
        论文 Eq.13:
        L_KD = KL(softmax(F_teacher/T) || softmax(F_student/T)) * T^2
        
        注意: 论文使用teacher作为target (q), student作为输入 (p)
        KL(q || p) = sum(q * log(q/p))
        
        Args:
            student_features: [B, N, D_s] (D_s=768 for base)
            teacher_features: [B, N, D_t] (D_t=1024 for large)
            
        Returns:
            loss_kd: 标量损失
        """
        T = self.kd_temperature
        
        # 池化到 [B, D] 
        student_pooled = student_features.mean(dim=1)  # [B, D_s]
        teacher_pooled = teacher_features.mean(dim=1)  # [B, D_t]
        
        # 如果维度不匹配，投影student特征到teacher维度
        if self.kd_proj is not None:
            student_pooled = self.kd_proj(student_pooled)  # [B, D_t]
        else:
            # 如果没有投影层但维度不同，取较小维度进行对齐
            min_dim = min(student_pooled.shape[-1], teacher_pooled.shape[-1])
            student_pooled = student_pooled[..., :min_dim]
            teacher_pooled = teacher_pooled[..., :min_dim]
        
        # 计算软标签
        # student: log_softmax (用于KL散度的log(p))
        student_soft = F.log_softmax(student_pooled / T, dim=-1)
        # teacher: softmax (用于KL散度的q)
        teacher_soft = F.softmax(teacher_pooled / T, dim=-1).detach()  # 确保不计算teacher梯度
        
        # KL散度: KL(q || p) = sum(q * (log(q) - log(p)))
        # F.kl_div(log(p), q) = sum(q * (log(q) - log(p)))
        loss_kd = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)
        
        return loss_kd
    
    def _compute_kd_loss_logits(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        基于 token logits 的知识蒸馏损失 (论文 Eq.13)
        
        L_KD = T^2 * KL(softmax(τ_T/T) || softmax(τ_S/T))
        
        Args:
            student_logits: [B, seq_len, vocab_size]
            teacher_logits: [B, seq_len, vocab_size]
            labels: [B, seq_len] (-100 表示 padding)
        """
        T = self.kd_temperature
        
        if labels is not None:
            # 只在有效 token 位置计算 KD loss
            valid_mask = (labels != -100)
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=student_logits.device)
            student_flat = student_logits[valid_mask]  # [N_valid, vocab_size]
            teacher_flat = teacher_logits[valid_mask]   # [N_valid, vocab_size]
        else:
            student_flat = student_logits.reshape(-1, student_logits.size(-1))
            teacher_flat = teacher_logits.reshape(-1, teacher_logits.size(-1))
        
        if student_flat.numel() == 0:
            return torch.tensor(0.0, device=student_logits.device)
        
        student_soft = F.log_softmax(student_flat / T, dim=-1)
        teacher_soft = F.softmax(teacher_flat / T, dim=-1).detach()
        
        loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)
        return loss
    
    @staticmethod
    def trajectory_to_text(trajectory) -> list:
        """
        将轨迹张量转换为文本字符串 (用于 decoder labels)
        
        格式: "x1,y1;x2,y2;...;x45,y45" (2位小数)
        
        Args:
            trajectory: [B, T, 2] tensor/ndarray 或 [T, 2]
        Returns:
            texts: list of str
        """
        import numpy as np
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.detach().cpu().numpy()
        if trajectory.ndim == 2:
            trajectory = trajectory[np.newaxis, ...]
        
        texts = []
        for b in range(trajectory.shape[0]):
            points = []
            for t in range(trajectory.shape[1]):
                x, y = trajectory[b, t, 0], trajectory[b, t, 1]
                points.append(f"{x:.2f},{y:.2f}")
            texts.append(";".join(points))
        return texts
    
    @staticmethod
    def text_to_trajectory(texts, trajectory_length=45):
        """
        解析生成的文本为轨迹张量
        
        Args:
            texts: list of str ("x1,y1;x2,y2;...")
            trajectory_length: 目标轨迹长度
        Returns:
            trajectory: [B, T, 2] tensor
        """
        batch = []
        for text in texts:
            points = []
            if not text or not text.strip():
                batch.append([[0.0, 0.0]] * trajectory_length)
                continue
            
            parts = text.strip().split(";")
            for part in parts[:trajectory_length]:
                part = part.strip()
                if not part:
                    continue
                try:
                    xy = part.split(",")
                    if len(xy) >= 2:
                        x = float(xy[0].strip())
                        y = float(xy[1].strip())
                        points.append([x, y])
                except (ValueError, IndexError):
                    continue
            
            # 用最后一个有效点填充
            while len(points) < trajectory_length:
                last = points[-1] if points else [0.0, 0.0]
                points.append(list(last))
            
            batch.append(points[:trajectory_length])
        return torch.tensor(batch, dtype=torch.float32)
    
    def generate_trajectory(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512
    ) -> torch.Tensor:
        """
        推理模式: token 生成 → 解析为轨迹坐标
        
        对应论文 Algorithm 1:
        "Decode trajectory tokens: Obtain τ̂ = Decoder(f)"
        
        Args:
            pixel_values: [B, 3, H, 2W]
            input_ids: [B, L]
            attention_mask: [B, L]
            
        Returns:
            trajectory: [B, T, 2]
        """
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            generated_ids = self.student_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                # 注意: 不传 attention_mask!
                # Florence-2 内部会拼接 image tokens + text tokens 并自动生成正确的 mask
                # 外部传入的 attention_mask 仅覆盖 text tokens, 会导致尺寸不匹配
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                early_stopping=False,
            )
            
            # Florence-2 是 encoder-decoder 架构
            # generate() 返回的是 decoder 输出 (不含 encoder input_ids)
            # 直接解码即可, 无需截断
            generated_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            trajectory = self.text_to_trajectory(
                generated_texts, self.trajectory_length
            )
        
        if was_training:
            self.train()
        
        return trajectory.to(pixel_values.device)
    
    def predict(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        推理模式: MLP 轨迹头直接预测
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            student_features = self.encode_multimodal(
                self.student_model, pixel_values, input_ids, attention_mask
            )
            trajectory_pred = self.trajectory_head(
                student_features['fusion_features'], attention_mask=None
            )
        if was_training:
            self.train()
        return trajectory_pred
    
    def get_trainable_parameters(self, base_lr: float = 1e-6) -> List[Dict]:
        """
        获取可训练参数组 (不包括frozen teacher)

        3层差分学习率:
        - vision_tower: base_lr * 0.1 (1e-7, 预训练视觉特征需极小lr)
        - other backbone: base_lr (5e-6, 预训练语言/融合模块)
        - 新增模块 (trajectory_head, alignment, kd_proj): base_lr * 50 (2.5e-4)
        """
        vision_lr = base_lr * 0.1   # 5e-7
        head_lr = base_lr * 50      # 2.5e-4

        # 分离 vision_tower 和其余 student 参数
        vision_params = list(self.student_model.vision_tower.parameters())
        vision_param_ids = set(id(p) for p in vision_params)
        other_student_params = [p for p in self.student_model.parameters()
                                if id(p) not in vision_param_ids]

        param_groups = [
            {'params': vision_params, 'lr': vision_lr},
            {'params': other_student_params, 'lr': base_lr},
            {'params': list(self.trajectory_head.parameters()), 'lr': head_lr},
        ]

        if self.alignment_module is not None:
            param_groups.append(
                {'params': list(self.alignment_module.parameters()), 'lr': head_lr}
            )

        if self.kd_proj is not None:
            param_groups.append(
                {'params': list(self.kd_proj.parameters()), 'lr': base_lr}
            )

        return param_groups
    
    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, **kwargs):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': {
                'student_model': self.student_model.state_dict(),
                'trajectory_head': self.trajectory_head.state_dict(),
            }
        }
        
        if self.alignment_module is not None:
            checkpoint['model_state_dict']['alignment_module'] = self.alignment_module.state_dict()
        
        if self.kd_proj is not None:
            checkpoint['model_state_dict']['kd_proj'] = self.kd_proj.state_dict()
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = False):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.student_model.load_state_dict(checkpoint['model_state_dict']['student_model'])
        self.trajectory_head.load_state_dict(checkpoint['model_state_dict']['trajectory_head'])
        
        if 'alignment_module' in checkpoint['model_state_dict'] and self.alignment_module is not None:
            self.alignment_module.load_state_dict(checkpoint['model_state_dict']['alignment_module'])
        
        if 'kd_proj' in checkpoint['model_state_dict'] and self.kd_proj is not None:
            self.kd_proj.load_state_dict(checkpoint['model_state_dict']['kd_proj'])
        
        print(f"Checkpoint loaded from {path}, epoch {checkpoint.get('epoch', 0)}")
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            return checkpoint['optimizer_state_dict']
        
        return None


class V2XVLMSimple(nn.Module):
    """
    简化版V2X-VLM (用于快速验证)
    
    不使用Florence-2, 仅用ResNet + Transformer
    """
    
    def __init__(
        self,
        trajectory_length: int = 45,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4
    ):
        super().__init__()
        
        # 简单的CNN视觉编码器
        from torchvision.models import resnet34, ResNet34_Weights
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # 特征投影
        self.vision_proj = nn.Conv2d(512, hidden_dim, 1)
        
        # Transformer融合
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 轨迹头
        self.trajectory_head = TrajectoryHead(
            hidden_dim=hidden_dim,
            trajectory_length=trajectory_length
        )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        trajectory_gt: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # 视觉编码
        vision_features = self.vision_encoder(pixel_values)  # [B, 512, H, W]
        vision_features = self.vision_proj(vision_features)  # [B, D, H, W]
        
        # 展平空间维度
        B, D, H, W = vision_features.shape
        vision_features = vision_features.flatten(2).permute(0, 2, 1)  # [B, H*W, D]
        
        # Transformer融合
        fusion_features = self.transformer(vision_features)
        
        # 轨迹预测
        trajectory_pred = self.trajectory_head(fusion_features)
        
        outputs = {'trajectory_pred': trajectory_pred}
        
        if trajectory_gt is not None:
            loss_traj = F.l1_loss(trajectory_pred, trajectory_gt)
            outputs['losses'] = {'loss_traj': loss_traj}
        
        return outputs
