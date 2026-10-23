
package com.kartik.rag.service;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import com.fasterxml.jackson.databind.ObjectMapper;

@Service
public class KgExtractionService {

    private static final Logger log =
            LoggerFactory.getLogger(KgExtractionService.class);

    private final WebClient ollamaWebClient;
    private final String chatModel;
    private final ObjectMapper objectMapper;

    public KgExtractionService(
            @Qualifier("ollamaWebClient") WebClient ollamaWebClient,
            @Value("${ollama.chat-model:llama3.2:3b}") String chatModel,
            ObjectMapper objectMapper) {

        this.ollamaWebClient = ollamaWebClient;
        this.chatModel = chatModel;
        this.objectMapper = objectMapper;
    }

    public ExtractionResult extract(List<ChunkRow> chunks) {
        if (chunks == null || chunks.isEmpty()) {
            return emptyResult();
        }

        log.info("Starting KG extraction on {} chunks", chunks.size());

        StringBuilder sb = new StringBuilder();

        chunks.forEach(c -> sb.append("[chunk ")
                .append(c.chunkIndex())
                .append("] ")
                .append(c.content())
                .append("\n\n"));

        String prompt = """
                You are an information extraction system.
                Read the provided chunks and extract structured knowledge graph data.

                Output ONLY valid JSON in this exact shape:
                {
                  "entities":[{"name":"","canonicalKey":"","entityType":"","description":"","aliases":[""]}],
                  "events":[{"eventType":"","eventCategory":"","name":"","chapter":"","location":"","startYear":null,"endYear":null}],
                  "participants":[{"eventName":"","actorName":"","role":"","outcome":"","chunkIndex":0}],
                  "relations":[{"subjectName":"","predicate":"","objectName":"","objectText":"","chunkIndex":0}]
                }

                Guidelines:
                - Use lowercase with underscores for canonicalKey (e.g., "cao_cao").
                - eventType is a short label like "battle", "incident", or "visit".
                - Link participants to events by matching eventName and actorName.
                - chunkIndex must reference the chunk number where the fact appears.
                - Do not invent facts that are absent from the text.
                - Use empty arrays when no items of a type can be extracted.
                - Use null for unknown years.
                - Return JSON only, without markdown fences or explanations.

                Text:
                """ + sb;

        OllamaChatRequest request = new OllamaChatRequest(
                chatModel,
                List.of(
                        new ChatMessage(
                                "system",
                                "Extract structured entities, events, participants, "
                                        + "and relations from text. Return valid JSON only."),
                        new ChatMessage("user", prompt)),
                false,
                new ChatOptions(0.0, "json"));

        try {
            OllamaChatResponse response = ollamaWebClient.post()
                    .uri("/api/chat")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(request)
                    .retrieve()
                    .bodyToMono(OllamaChatResponse.class)
                    .block();

            if (response == null
                    || response.message() == null
                    || response.message().content() == null
                    || response.message().content().isBlank()) {

                log.warn("KG extraction returned empty response");
                return emptyResult();
            }

            ExtractionResult result =
                    parseContent(response.message().content());

            log.info(
                    "KG extraction parsed entities={} events={} participants={} relations={}",
                    safeSize(result.entities()),
                    safeSize(result.events()),
                    safeSize(result.participants()),
                    safeSize(result.relations()));

            return result;

        } catch (Exception e) {
            log.error("KG extraction failed: {}", e.getMessage());
            return emptyResult();
        }
    }

    private ExtractionResult emptyResult() {
        return new ExtractionResult(
                List.of(),
                List.of(),
                List.of(),
                List.of());
    }

    public record ChunkRow(
            java.util.UUID id,
            int chunkIndex,
            String content) {
    }

    public record ExtractionResult(
            List<ExtractedEntity> entities,
            List<ExtractedEvent> events,
            List<ExtractedParticipant> participants,
            List<ExtractedRelation> relations) {
    }

    public record ExtractedEntity(
            String name,
            String canonicalKey,
            String entityType,
            String description,
            List<String> aliases) {
    }

    public record ExtractedEvent(
            String eventType,
            String eventCategory,
            String name,
            String chapter,
            String location,
            Integer startYear,
            Integer endYear) {
    }

    public record ExtractedParticipant(
            String eventName,
            String actorName,
            String role,
            String outcome,
            Integer chunkIndex) {
    }

    public record ExtractedRelation(
            String subjectName,
            String predicate,
            String objectName,
            String objectText,
            Integer chunkIndex) {
    }

    private int safeSize(List<?> list) {
        return list == null ? 0 : list.size();
    }

    private ExtractionResult parseContent(String content) {
        try {
            return objectMapper.readValue(content, ExtractionResult.class);
        } catch (Exception e) {
            log.warn(
                    "Primary parse failed; attempting fallback JSON extraction. Content snippet: {}",
                    snippet(content));

            String fallback = extractJson(content);

            if (fallback == null) {
                log.error("KG extraction content could not be parsed as JSON");
                return emptyResult();
            }

            try {
                return objectMapper.readValue(fallback, ExtractionResult.class);
            } catch (Exception ex) {
                log.error("Fallback parse failed: {}", ex.getMessage());
                return emptyResult();
            }
        }
    }

    private String extractJson(String text) {
        int start = text.indexOf('{');
        int end = text.lastIndexOf('}');

        if (start >= 0 && end > start) {
            return text.substring(start, end + 1);
        }

        return null;
    }

    private String snippet(String text) {
        if (text == null) {
            return "";
        }

        return text.length() > 500
                ? text.substring(0, 500) + "..."
                : text;
    }

    private record OllamaChatRequest(
            String model,
            List<ChatMessage> messages,
            boolean stream,
            ChatOptions options,
            String format) {

        OllamaChatRequest(
                String model,
                List<ChatMessage> messages,
                boolean stream,
                ChatOptions options) {
            this(model, messages, stream, options, options.format());
        }
    }

    private record ChatOptions(double temperature, String format) {
    }

    private record ChatMessage(String role, String content) {
    }

    private record OllamaChatResponse(OllamaMessage message) {
    }

    private record OllamaMessage(String role, String content) {
    }
}
