```mermaid
flowchart TD
A[CSV: data/selected_questions.csv] --> B[main()]
B --> B1[检查/设置环境变量]
B --> B2[Ollama /api/tags 探测]
B2 -->|无模型| B3[自动启动 ollama serve + 尝试 register_local_manifests(ollama pull)]
B --> B4[冒烟测试: llama3.2:latest]
B --> C[generate_all_answers()]
C --> D[RealModelAnswerGenerator]

subgraph RAG 与生成
D --> E[检索入口: get_retrieved_passages / get_retrieved_context]
E --> F{向量检索可用?}
F -->|是| G[SentenceTransformer 编码查询]
G --> H[Chroma Collection 'braincheck' 查询 (documents + distances)]
H --> I[距离 → 分数(≈1 - distance); 文档截断]
F -->|否| J{RAG_ENABLE_LOAD_DATA==1?}
J -->|是| K[加载 braincheck_knowledge_base.pkl]
K --> L[SimpleBrainCheckLoader 关键词检索]
L --> M[赋予基线分数(0.45↓)]
J -->|否| N[无检索: 空上下文]

I --> O[融合: 向量分数 + 关键词基线分数(取最大)]
M --> O
O --> P[排序取 top_k; 产出 passages/scores/raw_distances]
P --> Q[编号拼接为 Contexts: [1]..[k]]

Q --> R[构造证据优先 Prompt]
R --> S[/api/generate (Ollama)]
S --> T[答案后处理: 超时/重试/词数截断]
end

C --> U[写出 CSV: results/real_answers_<timestamp>.csv]
B3 --> V[serve 日志: results/ollama_serve_<ts>.log]

```