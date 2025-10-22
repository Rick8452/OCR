# OCR Wallet Extractor

## Descripción general
OCR Wallet Extractor es un servicio basado en **FastAPI** y **Python 3.11** que permite extraer, validar y estructurar información desde documentos oficiales mexicanos como el **INE**, **CURP** y **Acta de Nacimiento**.  
El sistema utiliza **modelos OCR (Doctr / PP-OCRv5)** combinados con **plantillas de coordenadas (X,Y)** para identificar con precisión los campos dentro de los documentos.

## Características principales
- Extracción de texto mediante OCR de alto rendimiento.
- Uso de plantillas fijas por coordenadas para mayor precisión en documentos estandarizados.
- Validación automática de tipo de documento y consistencia de formato.
- Procesamiento de archivos PDF e imágenes.
- Almacenamiento de resultados JSON en **Amazon S3**.
- Endpoints API REST listos para integración con sistemas front-end.
- Compatible con contenedores **Docker** para despliegues productivos.

## Arquitectura
El proyecto se organiza en módulos:
- **services/**: contiene los motores OCR, extractores, validadores y clasificadores.
- **routers/**: define los endpoints FastAPI (como `/ocr/extract`).
- **storage/**: maneja la persistencia en S3.
- **main.py**: punto de entrada de la aplicación.
- **Dockerfile**: define la imagen de ejecución del servicio.

## Ejemplo de uso
### 1. Subir un documento
```bash
POST /ocr/extract
Form fields:
- usuarioID: "12345"
- tipo_documento: "ine"
- file: <archivo INE o PDF>
