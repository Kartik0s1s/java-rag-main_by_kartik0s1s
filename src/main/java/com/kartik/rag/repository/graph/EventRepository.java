package com.kartik.rag.repository.graph;

import org.springframework.data.jpa.repository.JpaRepository;

import com.kartik.rag.model.graph.EventEntity;

public interface EventRepository extends JpaRepository<EventEntity, Long> {
}
