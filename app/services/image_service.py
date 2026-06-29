import os
import shutil
import uuid

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


def save_image(file):

    filename = (
        f"{uuid.uuid4()}_{file.filename}"
    )

    filepath = os.path.join(
        UPLOAD_DIR,
        filename
    )

    with open(
        filepath,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    return filepath