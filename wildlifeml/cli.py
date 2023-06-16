"""Entry-point setups for the wildlifeml CLI."""
from typing import Optional

import click

from wildlifeml import MegaDetector


@click.command()
@click.option('--directory', '-d', help='Directory with images.', required=True)
@click.option('--output', '-o', help='Output file for results.')
@click.option('--batch_size', '-b', help='Batch size for processing.', default=1)
@click.option(
    '--confidence_threshold',
    help='Confidence threshold for including bounding box in results.',
    default=0.1,
)
def get_bbox(
    directory: str,
    output: Optional[str],
    batch_size: int,
    confidence_threshold,
) -> None:
    """Produce bounding boxes for a directory filled with images."""
    md = MegaDetector(batch_size=batch_size, confidence_threshold=confidence_threshold)
    md.predict_directory(directory=directory, output_file=output)
