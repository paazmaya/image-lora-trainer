from PIL import Image
from src.preprocess import ImagePreprocessor


def test_center_crop_and_resize():
    preprocessor = ImagePreprocessor(resolution=256, crop_focus=None)

    # Create dummy image
    img = Image.new("RGB", (500, 300), color="red")

    processed = preprocessor.process_image(img)

    assert processed.size == (256, 256)


def test_process_folder(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create dummy images
    Image.new("RGB", (100, 100)).save(input_dir / "test1.jpg")
    Image.new("RGB", (100, 100)).save(input_dir / "test2.png")

    preprocessor = ImagePreprocessor(resolution=64)
    stats = preprocessor.process_folder(input_dir, output_dir)

    assert len(stats["trained"]) == 2
    assert len(stats["skipped"]) == 0
    assert len(stats["failed"]) == 0
    assert (output_dir / "test1.png").exists()
    assert (output_dir / "test2.png").exists()


def test_process_folder_skipping(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create dummy images
    Image.new("RGB", (100, 100)).save(input_dir / "test1.jpg")
    Image.new("RGB", (100, 100)).save(input_dir / "test2.png")

    preprocessor = ImagePreprocessor(resolution=64, crop_focus="person")

    # Mock _content_aware_crop to always return None (simulate no detection)
    monkeypatch.setattr(preprocessor, "_content_aware_crop", lambda x: None)
    # Mock model to be truthy so process_image uses content aware path
    preprocessor.model = True

    stats = preprocessor.process_folder(input_dir, output_dir)

    assert len(stats["trained"]) == 0
    assert len(stats["skipped"]) == 2
    assert len(stats["failed"]) == 0
