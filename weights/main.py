import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from my_funcs import (
    class_id_to_label, load_pt_model, load_sklearn_model, transform_image, load_bert, load_tokenizer#, clean
    )
import logging
from contextlib import asynccontextmanager
import PIL
import logging

logger = logging.getLogger('uvicorn.info')

class ImageResponse(BaseModel):
    class_name: str # dog, cat, etc
    class_index: int # class index from imagenet json file

class TextInput(BaseModel):
    text: str   # some user text to classify

class TextResponse(BaseModel):
    label: str  # tg post thematic
    prob: float # confidence


pt_model = None 
sk_model = None
bert_model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global pt_model
    global sk_model
    global bert_model
    global tokenizer
    pt_model = load_pt_model()
    logger.info('Torch model loaded')
    sk_model = load_sklearn_model()
    logger.info('Sklearn model loaded')
    bert_model = load_bert()
    logger.info('Bert model loaded')
    tokenizer = load_tokenizer()
    logger.info('tokenizer downloaded')
    yield
    # Clean up the ML models and release the resources
    del pt_model, sk_model, bert_model, tokenizer


app = FastAPI(lifespan=lifespan)

@app.get('/')
def return_info():
    return 'Hello World!'

@app.post('/clf_image')
def classify_image(file: UploadFile):
    # open image
    image = PIL.Image.open(file.file) 
    # preprocess image
    adapted_image = transform_image(image) 
    # log 
    logger.info(f'{adapted_image.shape}')
    # predict 
    with torch.inference_mode():
        pred_index = pt_model(adapted_image).numpy().argmax()
    # convert index to class
    imagenet_class = class_id_to_label(pred_index)
    # make correct response
    response = ImageResponse(
        class_name=imagenet_class,
        class_index=pred_index
    )
    return response

@app.post('/clf_text')
def classify_text(data: TextInput):
    class_list = ['Crypto', 'Beauty', 'Sports', 'Tech', 'Finance']
    encoded_input = tokenizer(data.text, max_length=64, truncation=True, padding='max_length', return_tensors='pt')
    encoded_input = {k: v.to(bert_model.device) for k, v in encoded_input.items()}
    
    with torch.no_grad():
        model_output = bert_model(**encoded_input)

    embeddings = model_output.last_hidden_state[:, 0, :]

    embeddings = torch.nn.functional.normalize(embeddings)
        
    embeddings_np = embeddings.cpu().numpy()

    pred_class = sk_model.predict(embeddings_np)
        
    pred_proba = sk_model.predict_proba(embeddings_np)
    confidence = np.max(pred_proba)

    response = TextResponse(
        label=class_list[pred_class[0]],
        prob=confidence
    )
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)

