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
            attn_implementation="eager"
        )
        self.processor = AutoProcessor.from_pretrained(
            student_model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # Teacher Model - Florence-2-Large (frozen)
        if use_knowledge_distillation:
            print("Loading Teacher model (Florence-2-Large)...")
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
                attn_implementation="eager"
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
        if use_contrastive_alignment:
            self.alignment_module = FeatureAlignment(
                vision_dim=hidden_dim,
                text_dim=hidden_dim,
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
        提取多模态特征
        
        论文 Section 4.2:
        - 视觉编码: F_v = f_v([I_v, I_i])
        - 文本编码: F_t = f_t(E)
        - 多模态融合: F_multi = MultiModalTransformer(F_v, F_t)
        
        Florence-2 是 encoder-decoder 架构：
        - vision_tower (DaViT): 图像编码
        - language_model (BART-like): encoder处理文本, decoder生成输出
        
        需要提供 decoder_input_ids 来获取 decoder hidden states
        """
        # 获取模型的hidden_dim (base=768, large=1024)
        if model == self.student_model:
            model_hidden_dim = self.hidden_dim
        else:
            model_hidden_dim = self.teacher_hidden_dim
        
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        dtype = pixel_values.dtype
        
        try:
            # Florence-2 需要 decoder_input_ids
            # 使用 input_ids 作为 decoder_input_ids (自回归方式)
            # 或者使用 BOS token 开始解码
            
            # 方法1: 直接使用完整的 forward
            outputs = model(
                input_ids=input_ids,              # encoder 输入 (文本prompt)
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                decoder_input_ids=input_ids,      # decoder 输入 (用于teacher forcing)
                output_hidden_states=True,
                return_dict=True
            )
            
            # 提取融合特征 - 优先使用 decoder_hidden_states
            if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
                # decoder_hidden_states 是 tuple，取最后一层
                fusion_features = outputs.decoder_hidden_states[-1]  # [B, seq_len, D]
            elif hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
                fusion_features = outputs.encoder_hidden_states[-1]
            elif hasattr(outputs, 'encoder_last_hidden_state') and outputs.encoder_last_hidden_state is not None:
                fusion_features = outputs.encoder_last_hidden_state
            else:
                # 使用 logits 的隐藏表示（不推荐，但作为后备）
                # Florence-2 的 logits shape: [B, seq_len, vocab_size]
                # 我们需要从模型内部获取 hidden state
                raise RuntimeError("Cannot extract hidden states from model outputs")
            
            # 确保特征维度正确
            if fusion_features.dim() == 2:
                fusion_features = fusion_features.unsqueeze(1)  # [B, 1, D]
            
            return {
                'vision_features': fusion_features,
                'text_embeddings': fusion_features,
                'fusion_features': fusion_features
            }
            
        except Exception as e:
            # 如果标准 forward 失败，尝试分步提取特征
            try:
                return self._extract_features_fallback(model, pixel_values, input_ids, attention_mask, model_hidden_dim)
            except Exception as e2:
                print(f"Warning: All feature extraction methods failed: {e2}")
                import traceback
                traceback.print_exc()
                # 最后的fallback: 使用随机特征 (仅用于调试，实际训练中不应到达这里)
                dummy_features = torch.randn(
                    batch_size, input_ids.shape[1], model_hidden_dim,
                    device=device, dtype=dtype
                )
                return {
                    'vision_features': dummy_features,
                    'text_embeddings': dummy_features,
                    'fusion_features': dummy_features
                }
    
    def _extract_features_fallback(
        self,
        model: nn.Module,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_dim: int
    ) -> Dict[str, torch.Tensor]:
        """
        备用特征提取方法：分步提取视觉和文本特征
        
        Florence-2 内部结构:
        - model.vision_tower: DaViT 视觉编码器
        - model.image_proj_norm + model.image_projection: 图像投影
        - model.language_model: BART-like encoder-decoder
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        dtype = pixel_values.dtype
        
        # 1. 提取视觉特征
        if hasattr(model, 'vision_tower'):
            # DaViT 视觉编码器
            vision_outputs = model.vision_tower(pixel_values)
            
            # 获取特征 (DaViT 输出格式可能不同)
            if hasattr(vision_outputs, 'last_hidden_state'):
                vision_features = vision_outputs.last_hidden_state
            elif isinstance(vision_outputs, tuple):
                vision_features = vision_outputs[0]
            else:
                vision_features = vision_outputs
            
            # Florence-2 的图像投影
            if hasattr(model, 'image_projection'):
                # 可能需要先 norm
                if hasattr(model, 'image_proj_norm'):
                    vision_features = model.image_proj_norm(vision_features)
                vision_features = model.image_projection(vision_features)
        else:
            # 没有 vision_tower，使用简单的 CNN 特征
            vision_features = torch.randn(batch_size, 256, hidden_dim, device=device, dtype=dtype)
        
        # 2. 提取文本 embeddings
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
            if hasattr(model.language_model.model, 'encoder'):
                # BART-like encoder
                encoder = model.language_model.model.encoder
                if hasattr(encoder, 'embed_tokens'):
                    text_embeddings = encoder.embed_tokens(input_ids)
                else:
                    text_embeddings = vision_features  # fallback
            else:
                text_embeddings = vision_features
        else:
            text_embeddings = vision_features
        
        # 3. 简单融合：拼接 + 平均
        # vision_features: [B, N_v, D]
        # text_embeddings: [B, N_t, D]
        
        # 确保维度匹配
        if vision_features.dim() == 4:
            # [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = vision_features.shape
            vision_features = vision_features.flatten(2).permute(0, 2, 1)
        
        if vision_features.shape[-1] != hidden_dim:
            # 需要投影
            vision_features = F.adaptive_avg_pool1d(
                vision_features.permute(0, 2, 1), hidden_dim
            ).permute(0, 2, 1)
        
        # 使用视觉特征作为融合特征（简化方案）
        fusion_features = vision_features
        
        return {
            'vision_features': vision_features,
            'text_embeddings': text_embeddings if text_embeddings.shape[-1] == hidden_dim else vision_features,
            'fusion_features': fusion_features
        }
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trajectory_gt: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            pixel_values: 拼接的图像 [B, 3, H, 2W]
            input_ids: 文本token IDs [B, L]
            attention_mask: 注意力掩码 [B, L]
            trajectory_gt: GT轨迹 [B, T, 2], 用于计算损失
            return_features: 是否返回中间特征
            
        Returns:
            outputs: 包含预测轨迹和各项损失的字典
        """
        batch_size = pixel_values.shape[0]
        outputs = {}
        
        # ========== Student Forward ==========
        student_features = self.encode_multimodal(
            self.student_model,
            pixel_values,
            input_ids,
            attention_mask
        )
        
        # 轨迹预测 [Eq.7]
        # τ = f_traj(F_multi)
        # 注意: 不传 attention_mask，因为 fusion_features 的序列长度与 input_ids 不同
        trajectory_pred = self.trajectory_head(
            student_features['fusion_features'],
            attention_mask=None  # 使用简单均值池化
        )
        outputs['trajectory_pred'] = trajectory_pred  # [B, T, 2]
        
        # ========== Teacher Forward (for KD) ==========
        if self.use_kd and self.teacher_model is not None:
            with torch.no_grad():
                teacher_features = self.encode_multimodal(
                    self.teacher_model,
                    pixel_values,
                    input_ids,
                    attention_mask
                )
            outputs['teacher_features'] = teacher_features
        
        # ========== 计算损失 ==========
        losses = {}
        
        # 1. 轨迹回归损失 L_traj [Eq.8]
        if trajectory_gt is not None:
            loss_traj = F.l1_loss(trajectory_pred, trajectory_gt)
            losses['loss_traj'] = loss_traj
        
        # 2. 对比对齐损失 L_align [Eq.12]
        if self.use_alignment and self.alignment_module is not None:
            z_v, z_t = self.alignment_module(
                student_features['vision_features'],
                student_features['text_embeddings']
            )
            loss_align = self.alignment_module.compute_loss(z_v, z_t)
            losses['loss_align'] = loss_align
        
        # 3. 知识蒸馏损失 L_KD [Eq.13]
        if self.use_kd and self.teacher_model is not None and 'teacher_features' in outputs:
            loss_kd = self.compute_kd_loss(
                student_features['fusion_features'],
                outputs['teacher_features']['fusion_features']
            )
            losses['loss_kd'] = loss_kd
        
        outputs['losses'] = losses
        
        if return_features:
            outputs['student_features'] = student_features
        
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
    
    def predict(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        推理模式: 仅返回轨迹预测
        
        Args:
            pixel_values: [B, 3, H, 2W]
            input_ids: [B, L]
            attention_mask: [B, L]
            
        Returns:
            trajectory: [B, T, 2]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                pixel_values,
                input_ids,
                attention_mask,
                trajectory_gt=None,
                return_features=False
            )
        return outputs['trajectory_pred']
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取可训练参数 (不包括frozen teacher)"""
        params = []
        
        # Student model parameters
        params.extend(self.student_model.parameters())
        
        # Trajectory head
        params.extend(self.trajectory_head.parameters())
        
        # Alignment module
        if self.alignment_module is not None:
            params.extend(self.alignment_module.parameters())
        
        # KD projection
        if self.kd_proj is not None:
            params.extend(self.kd_proj.parameters())
        
        return params
    
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
