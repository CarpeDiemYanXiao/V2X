# src/models/losses.py
import torch
import torch.nn.functional as F

def trajectory_loss(pred_logits, gt_tokens, label_smoothing=0.05):
    """
    计算轨迹预测损失
    pred_logits: [B, 90, 1024]
    gt_tokens: [B, 45, 2] -> reshape后 [B*90]
    label_smoothing: 标签平滑系数，有助于训练稳定性（降低到0.05以提高学习速度）
    """
    # 动态获取词表大小
    vocab_size = pred_logits.shape[-1]
    
    # 展平并计算 CrossEntropy，忽略 -100 (Padding)
    # 添加label smoothing提高训练稳定性
    return F.cross_entropy(
        pred_logits.reshape(-1, vocab_size), 
        gt_tokens.reshape(-1), 
        ignore_index=-100,
        label_smoothing=label_smoothing
    )

def contrastive_loss(visual_features, text_features, temperature=0.07):
    # 归一化
    visual_features = F.normalize(visual_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # 计算相似度
    similarity = torch.matmul(visual_features, text_features.T) / temperature
    
    batch_size = visual_features.shape[0]
    labels = torch.arange(batch_size).to(visual_features.device)
    
    # 对称对比损失
    loss_v2t = F.cross_entropy(similarity, labels)
    loss_t2v = F.cross_entropy(similarity.T, labels)
    
    return (loss_v2t + loss_t2v) / 2

def kd_loss(student_logits, teacher_logits, temperature=1.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)