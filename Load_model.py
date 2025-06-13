

# 加载预训练模型
model = load_pretrained_model(
    model_path="models/power_system_transformer_model.pth",
    input_dim=x_train.shape[1],  # 输入维度需与训练时一致
    output_dim=y_train.shape[1]  # 输出维度需与训练时一致
)

# 运行推理
def predict(model, input_data):
    """使用加载的模型进行预测"""
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_data)
        outputs = model(input_tensor)
        preds = (outputs >= 0.5).float()
    return preds.numpy()

# 示例使用
# input_data = [...]  # 你的输入数据
# predictions = predict(model, input_data)