import requests  
import numpy as np  
from typing import List, Dict, Any  
import json  

class EmbeddingReranker:  
    def __init__(self, api_url: str):  
        self.api_url = api_url  
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:  
        """获取文本的嵌入向量"""  
        # 如果输入是单个字符串，转换为列表  
        if isinstance(texts, str):  
            texts = [texts]  
            
        payload = {  
            "input": texts,  
            "model": "bge-m3"  # 固定使用这个模型  
        }  
        
        try:  
            response = requests.post(self.api_url, json=payload)  
            response.raise_for_status()  
            
            response_data = response.json()  
            
            # 从响应中提取embeddings  
            embeddings = np.array([item["embedding"] for item in response_data["data"]])  
            return embeddings  
            
        except requests.exceptions.RequestException as e:  
            print(f"Request failed: {str(e)}")  
            raise  
        except json.JSONDecodeError as e:  
            print(f"Failed to parse JSON response: {str(e)}")  
            print(f"Raw response: {response.text}")  
            raise  
        except Exception as e:  
            print(f"Unexpected error: {str(e)}")  
            raise  
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:  
        """L2归一化"""  
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)  
        return embeddings / norms  
    
    def rerank(self, query: str, candidates: List[str], top_k: int = 3) -> List[Dict[str, Any]]:  
        """对候选文档进行重排序"""  
        # 获取query的embedding  
        query_embedding = self.get_embeddings([query])  
        query_embedding = self.normalize_embeddings(query_embedding)  
        
        # 获取候选文档的embeddings  
        candidate_embeddings = self.get_embeddings(candidates)  
        candidate_embeddings = self.normalize_embeddings(candidate_embeddings)  
        
        # 计算相似度  
        similarities = np.dot(query_embedding, candidate_embeddings.T)[0]  
        
        # 获取排序后的索引  
        ranked_indices = np.argsort(similarities)[::-1]  
        
        # 构建结果  
        results = []  
        for idx in ranked_indices[:top_k]:  
            results.append({  
                'text': candidates[idx],  
                'score': float(similarities[idx])  
            })  
        
        return results  


# 使用示例  
def main():  
    # 配置参数  
    API_URL = "http://127.0.0.1:12347/v1/embed"  
    
    # 初始化reranker  
    
    reranker = EmbeddingReranker(API_URL)  
    
    # 示例数据  
    query = "我想买一部手机"  
    candidates = [  
        "最新款iPhone性能评测",  
        "华为手机最新发布会",  
        "如何选择适合自己的手机",  
        "手机市场分析报告",  
        "电脑配件大全" , 
        "今天天气不错" ,
        "最好买个电池耐用耗电慢的手机"
    ]  
    
    try:  
        results = reranker.rerank(query, candidates, top_k=3)  
        
        # 打印结果  
        print(f"\nQuery: {query}\n")  
        for i, result in enumerate(results, 1):  
            print(f"Rank {i}:")  
            print(f"Text: {result['text']}")  
            print(f"Score: {result['score']:.4f}")  
            print()  
            
    except Exception as e:  
        print(f"Error occurred: {str(e)}")  
        import traceback  
        print(traceback.format_exc())  

if __name__ == "__main__":  
    main()  
