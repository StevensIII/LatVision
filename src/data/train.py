# src/data/train.py
import argparse
from ultralytics import YOLO
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import os
import torch

def get_optimal_device():
    """Detecta automáticamente el mejor dispositivo (GPU si disponible, CPU si no)"""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU detectada: {gpu_name} ({gpu_memory:.1f}GB) - Aceleración activada")
    else:
        device = 'cpu'
        print("💻 Usando CPU (normal para desarrollo)")
    return device

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento YOLOv8 con MLflow tracking")
    parser.add_argument("--model", type=str, required=True, help="Ruta del modelo base YOLOv8 (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Ruta al archivo data.yaml del dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--run_name", type=str, default="EcoVision_Run", help="Nombre del run en MLflow")
    parser.add_argument("--project", type=str, default="./runs", help="Ruta de la carpeta de proyecto para guardar los resultados")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Detectar dispositivo óptimo automáticamente
    device = get_optimal_device()

    mlflow.set_tracking_uri("file:./mlruns")  # Tracking local
    mlflow.set_experiment("EcoVision Nuevo")        # Elegir experimento
    with mlflow.start_run(run_name=args.run_name) as run:
        # Log parámetros (incluyendo dispositivo usado)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("img_size", args.imgsz)
        mlflow.log_param("data", args.data)
        mlflow.log_param("device", device)

        # Cargar modelo base
        model = YOLO(args.model)

        # Entrenamiento con dispositivo óptimo
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            name=args.run_name,
            project=args.project,
            device=device,  # Usa GPU automáticamente si está disponible
        )

        # --- Log de métricas de manera segura ---
        try:
            if hasattr(results.box, "loss") and len(results.box.loss) > 0:
                mlflow.log_metric("box_loss", float(results.box.loss[-1]))
            if hasattr(results.seg, "loss") and len(results.seg.loss) > 0:
                mlflow.log_metric("seg_loss", float(results.seg.loss[-1]))
            if hasattr(results.box, "map") and isinstance(results.box.map, dict) and 0.5 in results.box.map:
                mlflow.log_metric("mAP50", float(results.box.map[0.5]))
        except Exception as e:
            print(f"⚠️ No se pudieron loggear algunas métricas: {e}")

        # Ruta del mejor peso
        best_weights_path = os.path.join(args.project, args.run_name, "weights", "best.pt")
        print(f"\nEntrenamiento finalizado. Run ID: {run.info.run_id}")

        if os.path.exists(best_weights_path):
            # Log del archivo best.pt
            mlflow.log_artifact(best_weights_path, artifact_path="weights")

            # --- Log y registro en Registry en un solo paso ---
            mlflow.pytorch.log_model(
                pytorch_model=model.model,
                artifact_path="weights_model",
                registered_model_name="EcoVisionModel"  # Registro automático
            )
            print("✅ Modelo loggeado y registrado correctamente en MLflow Registry")
        else:
            print("⚠️ best.pt no encontrado, no se loggeo en MLflow")

        # --- Gráfico de pérdidas ---
        plt.figure(figsize=(8,6))
        if hasattr(results.box, "loss"):
            plt.plot(results.box.loss, label="box_loss")
        if hasattr(results.seg, "loss"):
            plt.plot(results.seg.loss, label="seg_loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training Loss")
        loss_plot_path = os.path.join("loss_plot.png")
        plt.savefig(loss_plot_path)
        mlflow.log_artifact(loss_plot_path)
        plt.close()

    print(f"✅ Entrenamiento y logueo completados. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
