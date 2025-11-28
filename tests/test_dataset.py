from PIL import Image
from src.dataset import LocalImageDataset


class MockTokenizer:
    def __init__(self):
        self.model_max_length = 77

    def __call__(self, text, **kwargs):
        class MockOutput:
            def __init__(self):
                self.input_ids = [[1] * 77]  # Dummy input_ids

        return MockOutput()


def test_dataset(tmp_path):
    # Create dummy image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(img_path)

    tokenizer = MockTokenizer()
    dataset = LocalImageDataset(
        image_paths=[img_path],
        tokenizer=tokenizer,
        instance_prompt="a photo of a test",
        resolution=64,
    )

    assert len(dataset) == 1
    item = dataset[0]
    assert "pixel_values" in item
    assert "input_ids" in item
    assert item["pixel_values"].shape == (3, 64, 64)
