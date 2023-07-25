import uvicorn
from fastapi import APIRouter, FastAPI, Request, requests
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from library.dreambooth_folder_creation_gui import dreambooth_folder_preparation

from modules import errors
from modules.api import models

from lora_gui import train_model
import os
import base64
import uuid
import shutil

from fastapi import BackgroundTasks

import concurrent.futures


def api_middleware(app: FastAPI):
    rich_available = True
    try:
        import anyio  # importing just so it can be placed on silent list
        import starlette  # importing just so it can be placed on silent list
        from rich.console import Console
        console = Console()
    except Exception:
        rich_available = False

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        # do not print backtrace on known httpexceptions
        if not isinstance(e, HTTPException):
            message = f"API error: {request.method}: {request.url} {err}"
            if rich_available:
                print(message)
                console.print_exception(show_locals=True, max_frames=2, extra_lines=1, suppress=[
                                        anyio, starlette], word_wrap=False, width=min([console.width, 200]))
            else:
                errors.report(message, exc_info=True)
        return JSONResponse(status_code=vars(e).get('status_code', 500), content=jsonable_encoder(err))

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)


class Api:
    def __init__(self, app: FastAPI):
        self.router = APIRouter()
        self.app = app
        api_middleware(self.app)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
        self.futures = {}

        self.add_api_route("/loraapi/v1/training", self.create_train,
                           methods=["POST"], response_model=models.TrainResponse)
        self.add_api_route("/loraapi/v1/training/{id}", self.fetch_train_log,
                           methods=["GET"], response_model=models.TrainLogResponse)
        self.add_api_route("/loraapi/v1/training/{id}", self.check_train,
                           methods=["POST"], response_model=models.TrainInfoResponse)
        self.add_api_route("/loraapi/v1/training/{id}", self.terminate_train,
                           methods=["PUT"], response_model=models.TrainInfoResponse)
        self.add_api_route("/loraapi/v1/training/{id}", self.delete_training,
                           methods=["DELETE"], response_model=models.TrainInfoResponse)

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    async def check_train(self, id: str):
        future = self.futures.get(id)
        if not future:
            raise HTTPException(
                status_code=404, detail="Training process not found")

        if future.running():
            return models.TrainInfoResponse(title="Start LoRA training", description="Already running", info="Running")
        elif future.done():
            return models.TrainInfoResponse(title="Start LoRA training", description="Training already completed", info="Done")
        else:
            return models.TrainInfoResponse(title="Start LoRA training", description="Task is pending, will start shortly", info="Pending")

    def fetch_train_log(self, id: str):
        base_dir = os.path.abspath(os.curdir)
        log_dir = os.path.join(base_dir, 'train_data', id, 'log')
        log_file_path = os.path.join(log_dir, 'log.txt')

        if not os.path.isfile(log_file_path):
            raise HTTPException(status_code=404, detail="Log file not found")

        with open(log_file_path, 'r') as log_file:
            train_log = log_file.read()

        return models.TrainLogResponse(title="LoRA training log", description="Log fetched", info=train_log, train_id=id)

    def train_model_bg(self, save_dir, train_data_dir, reg_data_dir, model_path, instance_prompt, class_prompt, max_resolution):
        base_dir = os.path.abspath(os.curdir)

        dreambooth_folder_preparation(
            save_dir,
            40,
            instance_prompt,
            f"{reg_data_dir}/{class_prompt}",
            1,
            class_prompt,
            train_data_dir
        )

        shutil.rmtree(save_dir)

        # train_model(
        # )

        # Assuming "test1.safetensors" is generated in "train_data/model/"
        original_file_path = os.path.join(
            train_data_dir, "model", "test1.safetensors")

        # Extract the train_id from the train_data_dir
        train_id = os.path.basename(train_data_dir)

        # Destination path in "stable-diffusion-webui/models/lora/{train_id}/"
        dest_dir = os.path.join(
            base_dir, "..", "stable-diffusion-webui", "models", "lora", train_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest_file_path = os.path.join(dest_dir, "test1.safetensors")

        # Move the file
        shutil.move(original_file_path, dest_file_path)

    async def create_train(self, request: Request, background_tasks: BackgroundTasks):
        base_dir = os.path.abspath(os.curdir)

        # Generate a unique ID for this training process
        train_id = str(uuid.uuid4())

        data = await request.json()

        # Extract the train_images
        train_images = data.get('input', {}).get('train_images', [])

        # Specify the directory where you want to save the images

        save_dir = os.path.join(base_dir, 'train_images', train_id)

        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the base64 images as files
        for i, img_base64 in enumerate(train_images):
            img_data = base64.b64decode(img_base64)
            img_path = os.path.join(save_dir, f'image_{i}.jpg')

            with open(img_path, 'wb') as img_file:
                img_file.write(img_data)

        train_data_dir = os.path.join(base_dir, "train_data", train_id)
        reg_data_dir = os.path.join(base_dir, "..", "reg_data")
        max_resolution = "512,512"
        model_path = os.path.join(base_dir, "..", "stable-diffusion-webui",
                                  "models", "Stable-diffusion", "model.safetensors")
        instance_prompt = "ohwx"
        class_prompt = "man"

        future = self.executor.submit(self.train_model_bg, save_dir, train_data_dir,
                                      reg_data_dir, model_path, instance_prompt, class_prompt, max_resolution)
        self.futures[train_id] = future

        return models.TrainResponse(title="LoRA training", description="Completed", info="Success", train_id=train_id)

    def terminate_train(self, id: str):
        future = self.futures.get(id)
        if not future:
            raise HTTPException(
                status_code=404, detail="Training process not found")

        # Try to cancel the future
        if future.cancel():
            return models.TrainInfoResponse(title="Terminate LoRA training", description="Terminated", info="Success")
        else:
            raise HTTPException(
                status_code=500, detail="Failed to terminate training process")

    async def delete_training(self, id: str):
        base_dir = os.path.abspath(os.curdir)

        if id not in self.futures:
            raise HTTPException(
                status_code=404, detail="Training task not found")
        if not self.futures[id].cancelled():
            self.futures[id].cancel()

        # Remove task from executor
        del self.futures[id]

        # Remove training data directory
        train_data_dir = os.path.join(base_dir, "train_data", id)
        model_data_dir = os.path.join(
            base_dir, "..", "stable-diffusion-webui", "models", "lora", id)
        if os.path.exists(train_data_dir):
            shutil.rmtree(train_data_dir)

        if os.path.exists(model_data_dir):
            shutil.rmtree(model_data_dir)

        return models.TrainInfoResponse(title="Remove LoRA training", description="Deleted", info="Success")

    def launch(self, server_name, port):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name,
                    port=port, timeout_keep_alive=0)
