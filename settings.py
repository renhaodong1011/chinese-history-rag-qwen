"""
Created on 2026/1/5 by Renhaodong
Description:
"""
payload = {
  "model": "Qwen/Qwen2.5-32B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": ""
    }
  ],
  "stream": False,
  "max_tokens": 1024,
  "thinking_budget": 4096,
  "min_p": 0.05,
  "stop": None,
  "temperature": 0.7,
  "top_p": 0.7,
  "top_k": 50,
  "frequency_penalty": 0.5,
  "n": 1,
  "response_format": {
    "type": "text"
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "description": "<string>",
        "name": "<string>",
        "parameters": {},
        "strict": False
      }
    }
  ]
}
