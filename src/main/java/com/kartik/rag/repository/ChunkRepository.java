package com.kartik.rag.repository;

import java.util.List;
import java.util.UUID;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import com.kartik.rag.model.Chunk;

@Repository
public interface ChunkRepository extends CrudRepository<Chunk, UUID> {
    List<Chunk> findByDocumentId(UUID documentId);
}
