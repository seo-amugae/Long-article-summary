from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import torch 
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration 
from fastapi.responses import FileResponse

tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization').to('cuda') 

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class TextRequest(BaseModel):
    text: str


@app.get("/")
def get_root():
    return FileResponse("index.html")

@app.post("/summarize")
async def summarize(req: TextRequest):
    try:
        text = req.text.replace('\n', ' ')
        if len(text) < 100:
            return {"error": f"입력 내용이 너무 짧습니다. 100글자 이상 입력해주세요. (현재 {len(text)}글자)"}
        input_ids = tokenizer.encode(text)

        if len(input_ids) > 1024:
            return {
                "error": f"입력 내용이 너무 깁니다. 최대 1024 단어까지 지원됩니다. (현재 {len(input_ids)} 단어)"
            }

        ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]

        out = model.generate(
            torch.tensor([ids]).to(model.device),
            num_beams=10, 
            max_length=120, 
            min_length=50, 
            length_penalty=1.2, 
            repetition_penalty=2.5,
            no_repeat_ngram_size=3, 
            eos_token_id=tokenizer.eos_token_id 
        )

        return {"summary": tokenizer.decode(out.squeeze().tolist(), skip_special_tokens=True)}

    except Exception as e:
        return {"error": str(e)}
