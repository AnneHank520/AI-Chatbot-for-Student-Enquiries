# PDF文档检索API使用指南

本文档提供了我们PDF语义搜索和检索系统中可用API端点的全面指南。这些API支持PDF处理、语义搜索以及与大型语言模型(LLM)集成，实现文档问答功能。可以结合test.html的具体调用方式进行参考。

## 基础URL

所有API端点均可在以下地址访问：

```
http://localhost:5000/api
```

## API概览

| 端点 | 方法 | 描述 |
|------|------|------|
| `/status` | GET | 检查API状态和可用资源 |
| `/process-pdf` | POST | 上传并处理新的PDF文件 |
| `/reprocess-pdf` | POST | 重新处理已存在的PDF文件 |
| `/list-files` | GET | 列出上传目录中所有可用的PDF文件 |
| `/query` | POST | 对处理过的PDF执行语义搜索 |
| `/generate-answer` | POST | 基于检索内容生成LLM回答 |

## 详细API文档

### 检查API状态

**端点：** `/status`  
**方法：** GET

获取API的当前状态和已加载资源的信息。

**示例请求：**
```bash
curl -X GET http://localhost:5000/api/status
```

**示例响应：**
```json
{
  "status": "running",
  "vector_db_loaded": true,
  "model_loaded": true,
  "sentences_loaded": true,
  "sentences_count": 1963,
  "url_index_count": 56,
  "keywords_index_count": 1245,
  "dashscope_api_available": true
}
```

### 处理PDF

**端点：** `/process-pdf`  
**方法：** POST  
**内容类型：** `multipart/form-data`

上传并处理新的PDF文件，提取文本并创建向量索引。

**参数：**
- `file`：要上传并处理的PDF文件（必需）

**示例请求：**
```bash
curl -X POST http://localhost:5000/api/process-pdf \
  -F "file=@document.pdf"
```

**示例响应：**
```json
{
  "message": "PDF processed successfully",
  "sentences_count": 1963,
  "vector_db_path": "models/vector_index.bin",
  "sentences_path": "models/sentences.json",
  "url_count": 56,
  "keywords_count": 1245
}
```

### 重新处理PDF

**端点：** `/reprocess-pdf`  
**方法：** POST  
**内容类型：** `application/json`

重新处理上传目录中已存在的PDF文件。

**参数：**
- `filename`：要重新处理的PDF文件名（可选，如未提供则处理上传目录中的第一个PDF）

**示例请求：**
```bash
curl -X POST http://localhost:5000/api/reprocess-pdf \
  -H "Content-Type: application/json" \
  -d '{"filename": "document.pdf"}'
```

**示例响应：**
```json
{
  "message": "PDF reprocessed successfully",
  "filename": "document.pdf",
  "sentences_count": 1963,
  "vector_db_path": "models/vector_index.bin",
  "sentences_path": "models/sentences.json",
  "url_count": 56,
  "keywords_count": 1245
}
```

### 列出文件

**端点：** `/list-files`  
**方法：** GET

列出上传目录中所有可用的PDF文件。

**示例请求：**
```bash
curl -X GET http://localhost:5000/api/list-files
```

**示例响应：**
```json
{
  "files": ["document1.pdf", "document2.pdf", "guide.pdf"]
}
```

### 查询

**端点：** `/query`  
**方法：** POST  
**内容类型：** `application/json`

对处理过的PDF执行语义搜索，检索与给定查询最相关的内容。

**参数：**
- `query`：搜索查询文本（必需）
- `top_k`：要返回的结果数量（默认：12）
- `context_size`：要包含的上下文句子数量（默认：5）

**示例请求：**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "医院在哪里",
    "top_k": 5,
    "context_size": 2
  }'
```

**示例响应：**
```json
{
  "query": "医院在哪里",
  "processed_query": "医院",
  "keywords": ["医院"],
  "results": [
    {
      "sentence": "在紧急情况下，前往医院的急诊部门，医疗专业人员将评估您的情况。医院通常非常繁忙，所以您可能需要等待，因为情况更严重的患者会优先得到治疗。",
      "distance": 0.3566,
      "indices": [142, 141, 143],
      "urls": ["www.healthdirect.gov.au"],
      "has_keyword_match": true,
      "has_url_match": false
    },
    ...
  ],
  "prompt_data": {
    "text_content": ["...内容块1...", "...内容块2..."],
    "relevant_links": ["www.healthdirect.gov.au", "..."],
    "prompt_template": "# 系统指令\n您是一个智能信息助手...[完整提示]"
  }
}
```

### 生成回答

**端点：** `/generate-answer`  
**方法：** POST  
**内容类型：** `application/json`

检索查询相关内容并使用大型语言模型生成回答。

**参数：**
- `query`：搜索查询文本（必需）
- `top_k`：要包含在上下文中的结果数量（默认：12）
- `context_size`：要包含的上下文句子数量（默认：5）
- `model`：要使用的LLM模型（默认："qwen-plus"）

**示例请求：**
```bash
curl -X POST http://localhost:5000/api/generate-answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "医院在哪里",
    "top_k": 5,
    "context_size": 2,
    "model": "qwen-plus"
  }'
```

**示例响应：**
```json
{
  "query": "医院在哪里",
  "processed_query": "医院",
  "prompt_template": "# 系统指令\n您是一个智能信息助手...[完整提示]",
  "answer": "在紧急情况下，您可以通过以下方式找到医院...[完整回答]",
  "model": "qwen-plus",
  "content_blocks_count": 5,
  "relevant_links_count": 2
}
```

## 使用API

### 环境设置

在使用API之前，请确保：

1. API服务器正在`http://localhost:5000`上运行
2. 如果使用LLM功能，请设置环境变量`DASHSCOPE_API_KEY`，填入您的API密钥

### 处理流程

典型的使用流程：

1. 使用`/process-pdf`上传PDF文件
2. 使用`/status`检查状态，确保处理成功完成
3. 使用`/query`搜索相关内容
4. 使用`/generate-answer`获取基于内容的LLM生成回答

### 错误处理

所有API端点返回适当的HTTP状态码：
- `200`：请求成功
- `400`：错误请求（缺少参数，无效的文件格式）
- `404`：资源未找到
- `500`：服务器错误

错误将在响应正文中包含描述性错误消息：

```json
{
  "error": "上传目录中未找到PDF文件"
}
```

## 集成示例

### 前端JavaScript示例

```javascript
// 示例：处理PDF文件
async function processPDF(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch('http://localhost:5000/api/process-pdf', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('处理PDF时出错:', error);
    throw error;
  }
}

// 示例：搜索内容
async function searchQuery(query, topK = 12, contextSize = 5) {
  try {
    const response = await fetch('http://localhost:5000/api/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: query,
        top_k: topK,
        context_size: contextSize
      })
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('搜索查询时出错:', error);
    throw error;
  }
}

// 示例：生成回答
async function generateAnswer(query) {
  try {
    const response = await fetch('http://localhost:5000/api/generate-answer', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: query
      })
    });
    
    const data = await response.json();
    return data.answer;
  } catch (error) {
    console.error('生成回答时出错:', error);
    throw error;
  }
}
```
