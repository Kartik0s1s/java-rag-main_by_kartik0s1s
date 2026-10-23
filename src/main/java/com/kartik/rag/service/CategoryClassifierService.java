
package com.kartik.rag.service;

import java.util.List;
import java.util.Locale;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
public class CategoryClassifierService {

    private static final Logger log =
            LoggerFactory.getLogger(CategoryClassifierService.class);

    private final WebClient ollamaWebClient;
    private final String chatModel;
    private final String labelList;

    public CategoryClassifierService(
            @Qualifier("ollamaWebClient") WebClient ollamaWebClient,
            @Value("${ollama.chat-model:llama3.2:3b}") String chatModel,
            @Value("${category.labels:HISTORY,CHILDCARE,LEGAL,TECHNOLOGY,FINANCE,SCIENCE,HEALTH,LITERATURE,FICTION,BUSINESS,EDUCATION,OTHER}")
            String labelList) {

        this.ollamaWebClient = ollamaWebClient;
        this.chatModel = chatModel;
        this.labelList = labelList;
    }

    public String classify(String text, String fallback) {
        if (text == null || text.isBlank()) {
            return fallback;
        }

        String sample = text.length() > 2000
                ? text.substring(0, 2000)
                : text;

        OllamaChatRequest request = new OllamaChatRequest(
                chatModel,
                List.of(
                        new ChatMessage(
                                "system",
                                "Classify the document into exactly one of these labels: "
                                        + labelList
                                        + ". Reply with the single label only."),
                        new ChatMessage("user", sample)),
                false,
                new ChatOptions(0.0));

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
                return fallback;
            }

            String label = response.message().content();
            String result = normalize(label, fallback);

            log.info(
                    "Auto-classified document as {} from model label '{}'",
                    result,
                    label);

            return result;

        } catch (Exception e) {
            log.warn("Category classification failed: {}", e.getMessage());
            return fallback;
        }
    }

    private String normalize(String value, String fallback) {
        if (value == null || value.isBlank()) {
            return fallback;
        }

        String cleaned = value.trim()
                .replaceAll("^[`'\"\\s]+|[`'\"\\s]+$", "")
                .toLowerCase(Locale.ROOT);

        String[] labels = labelList.split(",");

        for (String label : labels) {
            if (cleaned.equals(label.trim().toLowerCase(Locale.ROOT))) {
                return label.trim().toLowerCase(Locale.ROOT);
            }
        }

        return fallback;
    }

    private record OllamaChatRequest(
            String model,
            List<ChatMessage> messages,
            boolean stream,
            ChatOptions options) {
    }

    private record ChatOptions(double temperature) {
    }

    private record ChatMessage(String role, String content) {
    }

    private record OllamaChatResponse(OllamaMessage message) {
    }

    private record OllamaMessage(String role, String content) {
    }
}
