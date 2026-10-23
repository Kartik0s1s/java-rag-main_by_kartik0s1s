
package com.kartik.rag.service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;

@Service
public class EmbeddingService {

    private static final Logger log =
            LoggerFactory.getLogger(EmbeddingService.class);

    private final WebClient ollamaWebClient;
    private final String embeddingModel;
    private final int batchSize;
    private final MeterRegistry meterRegistry;
    private final Timer embeddingTimer;
    private final Counter embeddingErrors;

    public EmbeddingService(
            WebClient.Builder webClientBuilder,
            @Value("${ollama.base-url:http://localhost:11434}")
            String baseUrl,
            @Value("${ollama.embedding-model:nomic-embed-text}")
            String embeddingModel,
            @Value("${ollama.embedding.batch-size:16}")
            int batchSize,
            MeterRegistry meterRegistry) {

        this.ollamaWebClient = webClientBuilder
                .baseUrl(baseUrl)
                .build();

        this.embeddingModel = embeddingModel;
        this.batchSize = batchSize > 0 ? batchSize : 1;
        this.meterRegistry = meterRegistry;
        this.embeddingTimer =
                meterRegistry.timer("rag.embedding.duration");
        this.embeddingErrors =
                meterRegistry.counter("rag.embedding.errors");
    }

    public float[] embed(String text) {
        List<float[]> result = embedBatch(List.of(text));

        return result.isEmpty() ? new float[0] : result.get(0);
    }

    public List<float[]> embedBatch(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return List.of();
        }

        List<float[]> results = new ArrayList<>(
                Collections.nCopies(texts.size(), null));

        List<String> pendingBatch = new ArrayList<>(batchSize);
        List<Integer> pendingIndexes = new ArrayList<>(batchSize);

        for (int i = 0; i < texts.size(); i++) {
            String text = texts.get(i);

            if (text == null || text.isBlank()) {
                results.set(i, new float[0]);
                continue;
            }

            pendingBatch.add(text);
            pendingIndexes.add(i);

            if (pendingBatch.size() == batchSize) {
                assignBatch(results, pendingBatch, pendingIndexes);
                pendingBatch.clear();
                pendingIndexes.clear();
            }
        }

        if (!pendingBatch.isEmpty()) {
            assignBatch(results, pendingBatch, pendingIndexes);
        }

        return results;
    }

    private void assignBatch(
            List<float[]> results,
            List<String> batchInputs,
            List<Integer> batchIndexes) {

        List<float[]> vectors = callEmbeddingApi(batchInputs);

        for (int i = 0; i < batchIndexes.size(); i++) {
            results.set(batchIndexes.get(i), vectors.get(i));
        }
    }

    private List<float[]> callEmbeddingApi(List<String> inputs) {
        Timer.Sample sample = Timer.start(meterRegistry);

        try {
            EmbeddingRequest request =
                    new EmbeddingRequest(embeddingModel, inputs);

            EmbeddingResponse response = ollamaWebClient.post()
                    .uri("/api/embed")
                    .bodyValue(request)
                    .retrieve()
                    .bodyToMono(EmbeddingResponse.class)
                    .block();

            if (response == null
                    || response.embeddings() == null
                    || response.embeddings().isEmpty()) {
                throw new EmbeddingException(
                        HttpStatus.BAD_GATEWAY,
                        "Ollama returned empty embedding data");
            }

            if (response.embeddings().size() != inputs.size()) {
                throw new EmbeddingException(
                        HttpStatus.BAD_GATEWAY,
                        "Ollama returned a mismatched embedding count");
            }

            List<float[]> vectors = new ArrayList<>(inputs.size());

            for (List<Double> embedding : response.embeddings()) {
                if (embedding == null || embedding.isEmpty()) {
                    throw new EmbeddingException(
                            HttpStatus.BAD_GATEWAY,
                            "Ollama returned an empty embedding vector");
                }

                if (embedding.size() != 768) {
                    throw new EmbeddingException(
                            HttpStatus.BAD_GATEWAY,
                            "Expected 768 dimensions but received "
                                    + embedding.size());
                }

                float[] vector = new float[embedding.size()];

                for (int i = 0; i < embedding.size(); i++) {
                    vector[i] = embedding.get(i).floatValue();
                }

                vectors.add(vector);
            }

            log.info("Received Ollama embeddings for batch size {}",
                    inputs.size());

            return vectors;

        } catch (WebClientResponseException e) {
            embeddingErrors.increment();
            log.error("Ollama embedding request failed: HTTP {}",
                    e.getStatusCode());

            throw new EmbeddingException(
                    HttpStatus.BAD_GATEWAY,
                    "Ollama embedding request failed: "
                            + e.getResponseBodyAsString());

        } catch (EmbeddingException e) {
            embeddingErrors.increment();
            log.error("Ollama embedding failed: {}", e.getMessage());
            throw e;

        } catch (RuntimeException e) {
            embeddingErrors.increment();
            log.error("Could not connect to Ollama: {}", e.getMessage());

            throw new EmbeddingException(
                    HttpStatus.SERVICE_UNAVAILABLE,
                    "Could not connect to local Ollama. "
                            + "Check that Ollama is running.");

        } finally {
            sample.stop(embeddingTimer);
        }
    }

    private record EmbeddingRequest(
            String model,
            List<String> input) {
    }

    private record EmbeddingResponse(
            List<List<Double>> embeddings) {
    }

    public static class EmbeddingException extends RuntimeException {
        private final HttpStatus status;

        public EmbeddingException(HttpStatus status, String message) {
            super(message);
            this.status = status;
        }

        public HttpStatus getStatus() {
            return status;
        }
    }

    /*
     * Temporary compatibility for other services that still reference
     * EmbeddingService.OpenAiException.
     */
    public static class OpenAiException extends RuntimeException {
        private final HttpStatus status;

        public OpenAiException(HttpStatus status, String message) {
            super(message);
            this.status = status;
        }

        public HttpStatus getStatus() {
            return status;
        }
    }
}
