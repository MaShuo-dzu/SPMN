import torch
import torch.nn.functional as F


def compute_cosine_similarity_matrix(compute_vector, vectors):
    compute_vector = compute_vector.unsqueeze(0)

    cosine_similarity = F.cosine_similarity(compute_vector, vectors, dim=1)

    return cosine_similarity


if __name__ == "__main__":
    compute_vector = torch.tensor([1.0, 2.0, 3.0])
    vectors = torch.tensor(
        [3.0, -20.0, 1.0])

    similarity_vector = compute_cosine_similarity_matrix(compute_vector, vectors)

    print(similarity_vector)