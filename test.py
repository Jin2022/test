import torch
import mindietorch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

BATCH_SIZE = 128
sentences = ["This is a sentence." for _ in range(BATCH_SIZE)]

with torch.no_grad():
    # load model
    model_id = '/models/bge-m3' # 注意将文件中的model_id修改为实际路径
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torchscript=True)
    model.eval()

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
    inputs['input_ids'] = inputs['input_ids'].to(torch.int32)
    inputs['attention_mask'] = inputs['attention_mask'].to(torch.int32)
    model = torch.jit.trace(model, [inputs['input_ids'], inputs['attention_mask']], strict=False)
    
    # compile
    MIN_SHAPE = (1, 1) # 元组内的元素分别代表mindietorch编译出的模型所能接受的最小输入batch_size及最小sequence_length，请按需调整
    MAX_SHAPE = (128, 8192)  # 元组内的元素分别代表mindietorch编译出的模型所能接受的最大输入batch_size及最大sequence_length，请按需调整
    dynamic_inputs = []
    dynamic_inputs.append(mindietorch.Input(min_shape=MIN_SHAPE, max_shape=MAX_SHAPE, dtype=inputs['input_ids'].dtype))
    dynamic_inputs.append(mindietorch.Input(min_shape=MIN_SHAPE, max_shape=MAX_SHAPE, dtype=inputs['attention_mask'].dtype))
    compiled_model = mindietorch.compile(
       model,
       inputs = dynamic_inputs,
       precision_policy = mindietorch.PrecisionPolicy.PREF_FP16,
       truncate_long_and_double=True,
       require_full_compilation=False,
       allow_tensor_replace_int=False,
       min_block_size=3,
       torch_executed_ops=[],
       #soc_version根据硬件型号填入，"xxxxx"与npu-smi info打屏信息中的'Name'字段的前五位一致
       soc_version="Ascend910B3",
       optimization_level=0
    )

    # save model
    compiled_model.save(model_id+"/compiled_model.pt")
    print('compiled model saved!')
