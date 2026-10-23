
package com.kartik.rag.service;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.kartik.rag.config.PredicateMappingProperties;
import com.kartik.rag.service.KnowledgeGraphService.KgCountAnswer;

@Service
public class QuestionRouterService {

    private final KnowledgeGraphService knowledgeGraphService;
    private final WebClient ollamaWebClient;
    private final String chatModel;
    private final PredicateMappingProperties predicateMappingProperties;
    private final ObjectMapper objectMapper;

    public QuestionRouterService(
            KnowledgeGraphService knowledgeGraphService,
            @Qualifier("ollamaWebClient") WebClient ollamaWebClient,
            PredicateMappingProperties predicateMappingProperties,
            ObjectMapper objectMapper,
            @Value("${ollama.chat-model:llama3.2:3b}") String chatModel) {

        this.knowledgeGraphService = knowledgeGraphService;
        this.ollamaWebClient = ollamaWebClient;
        this.predicateMappingProperties = predicateMappingProperties;
        this.objectMapper = objectMapper;
        this.chatModel = chatModel;
    }

    public Optional<KgRoutedAnswer> routeToKnowledgeGraph(String question) {
        if (question == null || question.isBlank()) {
            return Optional.empty();
        }

        return routeViaIntent(question.trim());
    }

    private Optional<KgRoutedAnswer> routeViaIntent(String question) {
        try {
            IntentPayload intent = callIntent(question);

            if (intent == null
                    || intent.intent() == null
                    || intent.confidence() < 0.4) {
                return Optional.empty();
            }

            String intentName = intent.intent().toLowerCase();

            if ("relation_count".equals(intentName)) {
                String subject = clean(intent.subject());
                String object = clean(intent.object());
                String predicate = clean(intent.predicate());

                List<String> predicates = mapPredicates(object, predicate);

                if (subject == null || subject.isBlank() || predicates.isEmpty()) {
                    return Optional.empty();
                }

                Optional<KgCountAnswer> answer =
                        knowledgeGraphService.countRelations(
                                subject,
                                predicates.get(0),
                                null);

                return answer.map(a -> new KgRoutedAnswer(a, null));
            }

        } catch (Exception e) {
            // If intent classification fails, use the normal vector-search path.
            return Optional.empty();
        }

        return Optional.empty();
    }

    private IntentPayload callIntent(String question) {
        String systemPrompt = """
                You are an intent classifier for a knowledge graph QA system.
                Return JSON only with these fields:
                intent (relation_count or none),
                subject, predicate, object, confidence (0-1).

                Example:
                {"intent":"relation_count","subject":"Cao Cao",
                 "predicate":"child","object":"children","confidence":0.8}

                If the question is not asking for a count of graph relations,
                return:
                {"intent":"none","subject":null,"predicate":null,
                 "object":null,"confidence":0.0}

                Do not include markdown fences or explanations.
                """;

        OllamaChatRequest request = new OllamaChatRequest(
                chatModel,
                List.of(
                        new ChatMessage("system", systemPrompt),
                        new ChatMessage("user", question)),
                false,
                new ChatOptions(0.0, true));

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
                    || response.message().content() == null) {
                return null;
            }

            String content = response.message().content().trim();

            // Defensive handling if a model adds JSON markdown fences.
            if (content.startsWith("```")) {
                content = content.replaceFirst("^```(?:json)?\\s*", "");
                content = content.replaceFirst("\\s*```$", "");
            }

            return objectMapper.readValue(content, IntentPayload.class);

        } catch (WebClientResponseException e) {
            throw new EmbeddingService.OpenAiException(
                    (HttpStatus) e.getStatusCode(),
                    e.getResponseBodyAsString());
        } catch (Exception e) {
            // Invalid model output should not stop normal RAG question answering.
            return null;
        }
    }

    private List<String> mapPredicates(String object, String predicate) {
        if (predicate != null && !predicate.isBlank()) {
            return List.of(predicate.toLowerCase());
        }

        String key = object == null ? "" : object.toLowerCase();

        if (predicateMappingProperties.getPredicates() != null) {
            for (var entry : predicateMappingProperties.getPredicates().entrySet()) {
                if (entry.getValue().stream()
                        .anyMatch(s -> key.contains(s.toLowerCase()))) {
                    return List.of(entry.getKey());
                }
            }
        }

        return key.isBlank() ? List.of() : List.of(key);
    }

    private String clean(String input) {
        if (input == null) {
            return null;
        }

        return input.trim()
                .replaceAll("[?。，、！!;；]+$", "")
                .trim();
    }

    public record KgRoutedAnswer(KgCountAnswer answer, String documentTitle) {
        public String scopeLabel() {
            if (documentTitle == null) {
                return "across all documents";
            }

            return "within \"" + documentTitle + "\"";
        }
    }

    private record OllamaChatRequest(
            String model,
            List<ChatMessage> messages,
            boolean stream,
            ChatOptions options) {
    }

    private record ChatOptions(
            double temperature,
            boolean format) {
    }

    private record ChatMessage(String role, String content) {
    }

    private record OllamaChatResponse(OllamaMessage message) {
    }

    private record OllamaMessage(String role, String content) {
    }

    private record IntentPayload(
            String intent,
            String subject,
            String predicate,
            String object,
            double confidence) {
    }
}
