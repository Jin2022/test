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

FORWARD_BASE_URL = "https://llm.bywin.cn:30663/api/aisp-smart-center/proxy-openai/v1/chat/completions"  
API_KEY = "your-api-key-here"  
FORWARD_HEADERS = {  
    "Authorization": f"Bearer {API_KEY}",  
    "Content-Type": "application/json"  
}  

class ThinkTagProcessor:  
    def __init__(self):  
        self.reasoning_tag_buffer = ""  
        self.reasoning_complete = False  
        self.max_buffer_size = 8  
        self.content_buffer = ""  # 用于存储<think>之前的内容  
        
    def process_chunk(self, content: str) -> tuple[str, Optional[str]]:  
        if self.reasoning_complete:  
            return content, None  
            
        # 检查是否包含<think>标签  
        if "<think>" in content:  
            parts = content.split("<think>", 1)  
            self.content_buffer += parts[0]  # 保存<think>之前的内容  
            content = parts[1]  # 只处理<think>后的内容  
            
        self.reasoning_tag_buffer += content  
        
        if len(self.reasoning_tag_buffer) > self.max_buffer_size:  
            excess = len(self.reasoning_tag_buffer) - self.max_buffer_size  
            self.reasoning_tag_buffer = self.reasoning_tag_buffer[excess:]  
            
        if "</think>" in self.reasoning_tag_buffer:  
            self.reasoning_complete = True  
            # 分离reasoning内容和普通内容  
            parts = self.reasoning_tag_buffer.split("</think>", 1)  
            reasoning = parts[0]  
            remaining = parts[1] if len(parts) > 1 else ""  
            
            # 返回累积的普通内容和推理内容  
            return self.content_buffer + remaining, reasoning  
            
        return None, content  

class StreamStateManager:  
    def __init__(self):  
        self.states = defaultdict(ThinkTagProcessor)  
        self.last_access = defaultdict(datetime.now)  
        self.cleanup_interval = timedelta(minutes=30)  
        
    def get_state(self, response_id: str) -> ThinkTagProcessor:  
        self.last_access[response_id] = datetime.now()  
        return self.states[response_id]  
    
    def cleanup_old_states(self):  
        current_time = datetime.now()  
        expired_ids = [  
            response_id for response_id, last_access in self.last_access.items()  
            if current_time - last_access > self.cleanup_interval  
        ]  
        for response_id in expired_ids:  
            del self.states[response_id]  
            del self.last_access[response_id]  

# 创建全局状态管理器  
stream_state_manager = StreamStateManager()  

class ContentProcessor:  
    @staticmethod  
    async def process_stream_chunk(chunk: str, processor: ThinkTagProcessor) -> str:  
        try:  
            if chunk.startswith('data: '):  
                payload = chunk[len('data: '):].strip()  
                if payload == '[DONE]':  
                    return 'data: [DONE]\n\n'  

                data = json.loads(payload)  
                if 'choices' not in data:  
                    return chunk  

                for choice in data.get('choices', []):  
                    delta = choice.get('delta', {})  
                    content = delta.get('content', '')  

                    # 使用新的处理器处理内容  
                    cleaned_content, reasoning = processor.process_chunk(content)  

                    # 更新delta  
                    if cleaned_content is not None:  
                        delta['content'] = cleaned_content  
                    if reasoning is not None:  
                        delta['reasoning_content'] = reasoning  

                    # 移除空content的delta  
                    if not delta.get('content') and not delta.get('reasoning_content'):  
                        choice.pop('delta', None)  

                return f"data: {json.dumps(data)}\n\n"  
            return chunk  
        except Exception as e:  
            return chunk  

    @staticmethod  
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
    
    @staticmethod  
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
    """处理流式响应"""  
    response_id = str(id(response))  
    processor = stream_state_manager.get_state(response_id)  
    
    try:  
        async for chunk in response.aiter_bytes():  
            chunk_str = chunk.decode('utf-8')  
            processed_chunk = await ContentProcessor.process_stream_chunk(chunk_str, processor)  
            if processed_chunk:  
                yield processed_chunk.encode('utf-8')  
    finally:  
        if response_id in stream_state_manager.states:  
            del stream_state_manager.states[response_id]  
            del stream_state_manager.last_access[response_id]  

@app.post("/v1/chat/completions")  
async def chat_completions(request_data: ChatRequest):  
    """处理chat completions请求"""  
    try:  
        url = f"{FORWARD_BASE_URL}"  
        
        async with httpx.AsyncClient() as client:  
            is_stream = request_data.dict().get('stream', False)  
            
            # 发送请求  
            response = await client.post(  
                url=url,  
                headers=FORWARD_HEADERS,  
                json=request_data.dict(),  
                timeout=60.0  
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
                    media_type='text/event-stream'  
                )  
            
            # 处理非流式响应  
            response_data = response.json()  
            processed_response = ContentProcessor.process_chat_response(response_data)  
            return JSONResponse(content=processed_response)  
            
    except Exception as e:  
        raise HTTPException(status_code=500, detail=str(e))  

if __name__ == "__main__":  
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8000)
