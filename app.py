from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import re
import logging
import traceback
from pydantic import BaseModel
from typing import AsyncGenerator, Tuple, List, Optional
from collections import defaultdict
import asyncio
from datetime import datetime, timedelta

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    model_config = {
        "extra": "allow"
    }
    pass

# API配置
FORWARD_BASE_URL = "http://10.0.13.1:1025/v1/chat/completions"
API_KEY = "your-api-key-here"
FORWARD_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

class ProcessingError(Exception):
    """自定义处理错误类"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class ThinkTagProcessor:
    def __init__(self):
        self.reasoning_tag_buffer = ""
        self.reasoning_complete = False
        self.max_buffer_size = 8
        self.content_buffer = ""
        
    def process_chunk(self, content: str) -> tuple[str, Optional[str]]:
        try:
            if self.reasoning_complete:
                return content, None
                
            if "<think>" in content:
                parts = content.split("<think>", 1)
                self.content_buffer += parts[0]
                content = parts[1]
                
            self.reasoning_tag_buffer += content
            
            if len(self.reasoning_tag_buffer) > self.max_buffer_size:
                excess = len(self.reasoning_tag_buffer) - self.max_buffer_size
                self.reasoning_tag_buffer = self.reasoning_tag_buffer[excess:]
                
            if "</think>" in self.reasoning_tag_buffer:
                self.reasoning_complete = True
                parts = self.reasoning_tag_buffer.split("</think>", 1)
                reasoning = parts[0]
                remaining = parts[1] if len(parts) > 1 else ""
                
                return self.content_buffer + remaining, reasoning
                
            return None, content
        except Exception as e:
            logger.error(f"Error in process_chunk: {str(e)}")
            logger.error(f"Content that caused error: {content}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise ProcessingError("Error processing chunk", {
                "content": content,
                "error": str(e),
                "buffer_state": {
                    "reasoning_tag_buffer": self.reasoning_tag_buffer,
                    "content_buffer": self.content_buffer
                }
            })

class StreamStateManager:
    def __init__(self):
        self.states = defaultdict(ThinkTagProcessor)
        self.last_access = defaultdict(datetime.now)
        self.cleanup_interval = timedelta(minutes=30)
        
    def get_state(self, response_id: str) -> ThinkTagProcessor:
        try:
            self.last_access[response_id] = datetime.now()
            return self.states[response_id]
        except Exception as e:
            logger.error(f"Error getting state for response_id {response_id}: {str(e)}")
            raise ProcessingError(f"Failed to get state for response", {
                "response_id": response_id,
                "error": str(e)
            })
    
    def cleanup_old_states(self):
        try:
            current_time = datetime.now()
            expired_ids = [
                response_id for response_id, last_access in self.last_access.items()
                if current_time - last_access > self.cleanup_interval
            ]
            for response_id in expired_ids:
                del self.states[response_id]
                del self.last_access[response_id]
            logger.info(f"Cleaned up {len(expired_ids)} expired states")
        except Exception as e:
            logger.error(f"Error during state cleanup: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")

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
                    logger.warning(f"Received data without 'choices': {data}")
                    return chunk

                for choice in data.get('choices', []):
                    delta = choice.get('delta', {})
                    content = delta.get('content', '')

                    cleaned_content, reasoning = processor.process_chunk(content)

                    if cleaned_content is not None:
                        delta['content'] = cleaned_content
                    if reasoning is not None:
                        delta['reasoning_content'] = reasoning

                    if not delta.get('content') and not delta.get('reasoning_content'):
                        choice.pop('delta', None)

                return f"data: {json.dumps(data)}\n\n"
            return chunk
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Problematic chunk: {chunk}")
            raise ProcessingError("Failed to decode JSON", {
                "chunk": chunk,
                "error": str(e)
            })
        except Exception as e:
            logger.error(f"Error processing stream chunk: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise ProcessingError("Stream processing error", {
                "chunk": chunk,
                "error": str(e)
            })

    @staticmethod
    def extract_think_content(content: str) -> Tuple[str, List[str]]:
        try:
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
        except Exception as e:
            logger.error(f"Error extracting think content: {str(e)}")
            logger.error(f"Content that caused error: {content}")
            raise ProcessingError("Failed to extract think content", {
                "content": content,
                "error": str(e)
            })
    
    @staticmethod
    def process_chat_response(response_data: dict) -> dict:
        try:
            if 'choices' in response_data:
                for choice in response_data['choices']:
                    if 'message' in choice:
                        message = choice['message']
                        if 'content' in message:
                            content = message['content']
                            think_pattern = r'<think>(.*?)</think>'
                            matches = re.findall(think_pattern, content, re.DOTALL)
                            cleaned_content = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
                            
                            message['content'] = cleaned_content
                            message['reasoning_content'] = matches[0] if matches else ''
                            
                            if 'role' not in message:
                                message['role'] = 'assistant'
            
            return response_data
        except Exception as e:
            logger.error(f"Error processing chat response: {str(e)}")
            logger.error(f"Response data that caused error: {response_data}")
            raise ProcessingError("Failed to process chat response", {
                "response_data": response_data,
                "error": str(e)
            })

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def stream_response(response: httpx.Response) -> AsyncGenerator[bytes, None]:
    response_id = str(id(response))
    processor = stream_state_manager.get_state(response_id)
    
    try:
        async for chunk in response.aiter_bytes():
            try:
                chunk_str = chunk.decode('utf-8')
                processed_chunk = await ContentProcessor.process_stream_chunk(chunk_str, processor)
                if processed_chunk:
                    yield processed_chunk.encode('utf-8')
            except Exception as e:
                logger.error(f"Error processing chunk in stream: {str(e)}")
                logger.error(f"Chunk that caused error: {chunk}")
                # 发送错误信息到客户端
                error_message = {
                    "error": {
                        "message": "Error processing stream chunk",
                        "details": str(e)
                    }
                }
                yield f"data: {json.dumps(error_message)}\n\n".encode('utf-8')
    finally:
        try:
            if response_id in stream_state_manager.states:
                del stream_state_manager.states[response_id]
                del stream_state_manager.last_access[response_id]
        except Exception as e:
            logger.error(f"Error cleaning up stream state: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(request_data: ChatRequest):
    try:
        url = f"{FORWARD_BASE_URL}"
        
        async with httpx.AsyncClient() as client:
            is_stream = request_data.dict().get('stream', False)
            
            try:
                response = await client.post(
                    url=url,
                    headers=FORWARD_HEADERS,
                    json=request_data.dict(),
                    timeout=60.0
                )
            except httpx.RequestError as e:
                logger.error(f"Request error: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "message": "Failed to forward request",
                        "error": str(e),
                        "url": url
                    }
                )
            
            if response.status_code != 200:
                error_content = response.json()
                logger.error(f"Upstream API error: {error_content}")
                return JSONResponse(
                    status_code=response.status_code,
                    content={
                        "error": error_content,
                        "upstream_status": response.status_code
                    }
                )
            
            if is_stream:
                return StreamingResponse(
                    stream_response(response),
                    media_type='text/event-stream'
                )
            
            response_data = response.json()
            processed_response = ContentProcessor.process_chat_response(response_data)
            return JSONResponse(content=processed_response)
            
    except Exception as e:
        logger.error(f"Unhandled error in chat_completions: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Internal server error",
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )

@app.on_event("startup")
async def startup_event():
    async def cleanup_task():
        while True:
            try:
                await asyncio.sleep(300)
                stream_state_manager.cleanup_old_states()
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
    
    asyncio.create_task(cleanup_task())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
