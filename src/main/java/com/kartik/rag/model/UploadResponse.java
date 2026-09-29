package com.kartik.rag.model;

import java.util.UUID;

public record UploadResponse(UUID documentId, int chunksStored) {
}
