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
| `/delete-pdf-content` | POST | 从向量数据库中删除特定PDF文件的内容 |
| `/list-files` | GET | 列出上传目录中所有可用的PDF文件 |
| `/query` | POST | 对处理过的PDF执行语义搜索 |
| `/generate-answer` | POST | 基于检索内容生成LLM回答 |
| `/extract-url-content` | POST | 提取URL网页内容 |
| `/add-url-to-index` | POST | 提取URL内容并添加到向量数据库 |

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

上传并处理新的PDF文件，提取文本并添加到向量索引。与之前版本不同，现在新上传的PDF内容会被添加到现有索引，而不是替换它。

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
  "message": "PDF processed successfully and added to existing index",
  "filename": "document.pdf",
  "sentences_added": 1963,
  "total_sentences": 3500,
  "vector_db_path": "models/vector_index.bin",
  "url_count": 56,
  "keywords_count": 1245
}
```

### 重新处理PDF

**端点：** `/reprocess-pdf`  
**方法：** POST  
**内容类型：** `application/json`

重新处理上传目录中已存在的PDF文件。默认会先删除该文件之前的内容，然后添加新处理的内容。

**参数：**
- `filename`：要重新处理的PDF文件名（可选，如未提供则处理上传目录中的第一个PDF）
- `remove_previous`：是否先删除该文件之前的内容（默认：true）

**示例请求：**
```bash
curl -X POST http://localhost:5000/api/reprocess-pdf \
  -H "Content-Type: application/json" \
  -d '{"filename": "document.pdf"}'
```

**示例响应：**
```json
{
  "message": "PDF reprocessed successfully and added to index",
  "filename": "document.pdf",
  "sentences_added": 1963,
  "total_sentences": 3450,
  "previous_entries_removed": 1920,
  "vector_db_path": "models/vector_index.bin",
  "url_count": 56,
  "keywords_count": 1245
}
```

### 删除PDF内容

**端点：** `/delete-pdf-content`  
**方法：** POST  
**内容类型：** `application/json`

从向量数据库中删除特定PDF文件的所有内容，但不删除文件本身。

**参数：**
- `filename`：要删除内容的PDF文件名（必需）

**示例请求：**
```bash
curl -X POST http://localhost:5000/api/delete-pdf-content \
  -H "Content-Type: application/json" \
  -d '{"filename": "document.pdf"}'
```

**示例响应：**
```json
{
  "message": "Content from file document.pdf deleted successfully",
  "indices_removed": 1963,
  "sentences_remaining": 1487,
  "url_count": 42,
  "keywords_count": 980
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

### 提取URL内容

**端点：** `/extract-url-content`  
**方法：** POST  
**内容类型：** `application/json`

提取指定URL的网页内容，包括标题和正文文本。

**参数：**
- `url`：要提取内容的URL（必需，必须以http://或https://开头）

**示例请求：**
```bash
curl -X POST http://localhost:5000/api/extract-url-content \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

**示例响应：**
```json
{
  "status": "success",
  "title": "Example Domain",
  "content": "Example Domain This domain is for use in illustrative examples in documents...",
  "url": "https://example.com"
}
```

### 添加URL到索引

**端点：** `/add-url-to-index`  
**方法：** POST  
**内容类型：** `application/json`

提取URL网页内容并将其添加到向量数据库中。

**参数：**
- `url`：要添加到索引的URL（必需，必须以http://或https://开头）

**示例请求：**
```bash
curl -X POST http://localhost:5000/api/add-url-to-index \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

**示例响应：**
```json
{
  "message": "URL content successfully added to index",
  "url": "https://example.com",
  "title": "Example Domain",
  "sentences_added": 45,
  "total_sentences": 3545,
  "keywords_count": 28
}
```

### 查询

**端点：** `/query`  
**方法：** POST  
**内容类型：** `application/json`

对处理过的内容执行语义搜索，检索与给定查询最相关的内容。可以搜索PDF和URL内容。

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

#### 基本流程
1. 使用`/process-pdf`上传PDF文件（内容会累积到向量数据库）
2. 使用`/status`检查状态，确保处理成功完成
3. 使用`/query`搜索相关内容
4. 使用`/generate-answer`获取基于内容的LLM生成回答

#### 内容管理流程
1. 使用`/list-files`查看已上传的PDF文件
2. 使用`/reprocess-pdf`更新特定PDF的内容（默认会先删除旧内容）
3. 使用`/delete-pdf-content`从数据库中删除特定PDF的内容（不会删除文件本身）
4. 使用`/extract-url-content`提取网页内容查看
5. 使用`/add-url-to-index`将网页内容添加到向量数据库中

### 重要变更说明

与之前版本的主要区别：

1. **增量处理** - 新的PDF文件会被添加到现有索引，而不是替换旧内容
2. **多源内容** - 支持同时从PDF和URL获取内容并建立索引
3. **内容管理** - 提供了更细粒度的内容管理功能，包括删除特定内容
4. **元数据支持** - 每个句子现在都带有来源信息，便于追踪内容来源

### 错误处理

所有API端点返回适当的HTTP状态码：
- `200`：请求成功
- `400`：错误请求（缺少参数，无效的文件格式或URL格式）
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

// 示例：删除PDF内容
async function deletePDFContent(filename) {
  try {
    const response = await fetch('http://localhost:5000/api/delete-pdf-content', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        filename: filename
      })
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('删除PDF内容时出错:', error);
    throw error;
  }
}

// 示例：添加URL到索引
async function addUrlToIndex(url) {
  try {
    const response = await fetch('http://localhost:5000/api/add-url-to-index', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        url: url
      })
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('添加URL到索引时出错:', error);
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
