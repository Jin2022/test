from fastapi import FastAPI, Request, HTTPException  
from fastapi.responses import JSONResponse, StreamingResponse  
from fastapi.middleware.cors import CORSMiddleware  
import httpx  
import json  
import re  
from pydantic import BaseModel  
from typing import AsyncGenerator, Tuple, List, Optional  
from collections import defaultdict  
import asyncio  
from datetime import datetime, timedelta  

class ChatRequest(BaseModel):  
    model_config = {  
        "extra": "allow"  # 允许额外字段  
    }  
    pass  

FORWARD_BASE_URL = "http://10.0.13.1:1025/v1/chat/completions"  
API_KEY = "your-api-key-here"  
FORWARD_HEADERS = {  
    "Authorization": f"Bearer {API_KEY}",  
    "Content-Type": "application/json"  
}  

from typing import Optional

class ThinkTagProcessor:  
    def __init__(self):  
        self.reasoning_complete = False  
        self.buffer = ""  
        
    def process_chunk(self, content: str) -> tuple[str, Optional[str]]:  
        # 已完成思考阶段直接返回  
        if self.reasoning_complete:  
            if len(self.buffer) > 0:  
                result = self.buffer + content  # 合并缓冲区和新内容  
                self.buffer = ""  
                return result, None  
            return content, None  
        
        # 累积内容到缓冲区  
        self.buffer += content  
        
        # 检查是否包含结束标记  
        think_pos = self.buffer.find("</think>")  
        if think_pos != -1:  
            self.reasoning_complete = True  
            reasoning = self.buffer[:think_pos]  
            self.buffer = self.buffer[think_pos + 8:] 
            return None, reasoning  
        

        if len(self.buffer) > 8:  
            reasoning = self.buffer[:-8]  
            self.buffer = self.buffer[-8:]  
            return None, reasoning  
        
        return None, None  
    
async def process_stream_chunk(payload: str, processor: ThinkTagProcessor) -> str: 
    try:  
        if payload == '[DONE]':  
            return '[DONE]'  
        
        data = json.loads(payload)  
        if 'choices' not in data:  
            return payload  

        for choice in data.get('choices', []):  
            delta = choice.get('delta', {})  
            content = delta.get('content', '')  

            # 使用新的处理器处理内容  
            cleaned_content, reasoning = processor.process_chunk(content)  
            
            # 更新delta  
            if cleaned_content is not None:  
                delta['content'] = cleaned_content  
                delta.pop('reasoning_content', None)
            if reasoning is not None:  
                delta['reasoning_content'] = reasoning  
                delta.pop('content', None)

            # 移除空content的delta  
            if cleaned_content is None and reasoning is None:  
                return None   

        return json.dumps(data,ensure_ascii=False)  
    except Exception as e:  
        return payload  
    
def extract_think_content(content: str) -> Tuple[str, List[str]]:  
    """提取内容中的think标签内容"""  
    think_pattern = r'<think>(.*?)</think>'  
    reasoning_contents = []  
    
    def replace_func(match):  
        think_content = match.group(1).strip()  
        think_content = re.sub(r'\n+', ' ', think_content)  
        think_content = ' '.join(think_content.split())  
        reasoning_contents.append(think_content)  
        return ""  
    
    cleaned_content = re.sub(think_pattern, replace_func, content, flags=re.DOTALL)  
    return cleaned_content.strip(), reasoning_contents  
    
def process_chat_response(response_data: dict) -> dict:  
    """  
    处理chat completion响应，提取最后一条assistant消息中</think>前的内容  
    
    Args:  
        response_data (dict): 原始响应数据  
        
    Returns:  
        dict: 处理后的响应数据  
    """  
    if not isinstance(response_data, dict):  
        raise ValueError("响应数据必须是字典类型")  
    
    choices = response_data.get('choices', [])  
    if not choices:  
        return response_data  
        
    last_message = choices[-1].get('message', {})  
    content = last_message.get('content', '')  
    
    # 查找</think>的位置  
    end_pos = content.find('</think>')  
    if end_pos != -1:  
        # 提取</think>之前的内容作为reasoning_content  
        reasoning_content = content[:end_pos]  
        # 如果reasoning_content以<think>开头，则移除  
        if reasoning_content.startswith('<think>'):  
            reasoning_content = reasoning_content[7:]  
        
        # 保留</think>后的内容  
        cleaned_content = content[end_pos + 8:].strip()  
        
        last_message['content'] = cleaned_content  
        last_message['reasoning_content'] = reasoning_content  
    
    return response_data  

app = FastAPI()  

app.add_middleware(  
    CORSMiddleware,  
    allow_origins=["*"],  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)  

async def stream_response(response: httpx.Response) -> AsyncGenerator[bytes, None]:  
    buffer = ""  
    # 匹配 data: 开头直到下一个 data: 或结尾  
    pattern = re.compile(r'data: (.*?)(?=\n\ndata: |\Z)', re.DOTALL)  
    processor= ThinkTagProcessor()
    async for chunk in response.aiter_bytes():  
        buffer += chunk.decode('utf-8')  
        
        matches = pattern.finditer(buffer)  
        last_end = 0  
        for match in matches:  
            message = match.group(1)
            last_end = match.end()  
            
            if message:  
                processed_chunk = await process_stream_chunk(message, processor)  
                if processed_chunk:  
                    formatted_chunk = f"data: {processed_chunk}\n\n"  
                    yield formatted_chunk.encode("utf-8")
        
        # 保留未处理完的数据  
        buffer = buffer[last_end:]  
        
        

@app.post("/v1/chat/completions")  
async def chat_completions(request_data: ChatRequest):  
    """处理chat completions请求"""  
    try:  
        url = f"{FORWARD_BASE_URL}"  
        
        async with httpx.AsyncClient() as client:  
            is_stream = request_data.dict().get('stream', False)  
            model = request_data.dict().get('model', "qwen")  
            model = model.replace("deepseek","qwen")
            # 发送请求  
            response = await client.post(  
                url=url,  
                headers=FORWARD_HEADERS,  
                json=request_data.dict(),  
                timeout=300.0  
            )  
            
            # 处理错误响应  
            if response.status_code != 200:  
                return JSONResponse(  
                    status_code=response.status_code,  
                    content=response.json()  
                )  
            
            # 处理流式响应  
            if is_stream:  
                return StreamingResponse(  
                    stream_response(response),  
                    media_type='text/event-stream',
                    headers={  
                        'Cache-Control': 'no-cache',  
                        'Connection': 'keep-alive',  
                        'X-Accel-Buffering': 'no'  # 禁用 Nginx 缓冲  
                    }  
                )  
            
            # 处理非流式响应  
            response_data = response.json()  
            processed_response = process_chat_response(response_data)  
            return JSONResponse(content=processed_response)  
            
    except Exception as e:  
        raise HTTPException(status_code=500, detail=str(e))  

if __name__ == "__main__":  
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8000)
