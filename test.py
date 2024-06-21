from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama2-70b",
  messages=[{"role":"user","content":"how can we substract 2 from 10?"}],
  temperature=0.5,
  top_p=1,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")


