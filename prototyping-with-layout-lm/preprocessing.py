from dataclasses import dataclass
from typing import List, Union


@dataclass
class BoundingBox:
    start_x: Union[float, int]
    start_y: Union[float, int]
    end_x: Union[float, int]
    end_y: Union[float, int]


@dataclass
class Document:
    name: str
    token_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]
    positions: List[BoundingBox]
    token_labels: Optional[List[int]] = None


def chunk_document(
        document: Document, chunk_size: int, overlap_ratio: float
) -> List[Document]:
    num_tokens = len(document.token_ids)
    overlapped_num_tokens = int(chunk_size * overlap_ratio)
    start, stop = 0, min(chunk_size, num_tokens)

    doc_chunks = []
    while True:

        chunk = Document(
            name=document.name,
            token_ids=document.token_ids[start:stop],
            token_type_ids=document.token_type_ids[start:stop],
            attention_mask=document.attention_mask[start:stop],
            positions=document.positions[start:stop],
            token_labels=document.token_labels[start:stop],
        )
        doc_chunks.append(chunk.add_special_tokens())

        if stop >= num_tokens:
            break
        start = stop - overlapped_num_tokens
        stop = start + chunk_size

    return doc_chunks