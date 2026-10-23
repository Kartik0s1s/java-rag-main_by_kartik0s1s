
package com.kartik.rag.service;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.kartik.rag.model.ChunkSearchResult;
import com.kartik.rag.model.QueryRequest;
import com.kartik.rag.model.QueryResponse;
import com.kartik.rag.model.QueryResponseSource;
import com.kartik.rag.service.KnowledgeGraphService.KgCountAnswer;
import com.kartik.rag.service.QuestionRouterService.KgRoutedAnswer;

import reactor.core.publisher.Flux;

@Service
public class QueryService {

    private static final Logger log =
            LoggerFactory.getLogger(QueryService.class);

    private final EmbeddingService embeddingService;
    private final VectorStoreService vectorStoreService;
    private final WebClient ollamaWebClient;
    private final String chatModel;
    private final QuestionRouterService questionRouterService;
    private final ObjectMapper objectMapper;

    public QueryService(
            EmbeddingService embeddingService,
            VectorStoreService vectorStoreService,
            @Qualifier("ollamaWebClient") WebClient ollamaWebClient,
            QuestionRouterService questionRouterService,
            ObjectMapper objectMapper,
            @Value("${ollama.chat-model:llama3.2:3b}") String chatModel) {

        this.embeddingService = embeddingService;
        this.vectorStoreService = vectorStoreService;
        this.ollamaWebClient = ollamaWebClient;
        this.chatModel = chatModel;
        this.questionRouterService = questionRouterService;
        this.objectMapper = objectMapper;
    }

    public QueryResponse answer(QueryRequest request) {
        Optional<QueryResponse> kgResponse = tryKnowledgeGraph(request);

        if (kgResponse.isPresent()) {
            log.info(
                    "Query routed to knowledge graph path (category={})",
                    request.category());
            return kgResponse.get();
        }

        float[] queryEmbedding = embeddingService.embed(request.query());

        List<ChunkSearchResult> matches =
                vectorStoreService.searchSimilar(
                        queryEmbedding,
                        request.resolvedTopK(),
                        request.category());

        if (matches.isEmpty()) {
            return new QueryResponse(
                    "I could not find relevant information in the knowledge base.",
                    List.of());
        }

        String prompt = buildPrompt(request.query(), matches);
        String answer = invokeChatCompletion(prompt);

        List<QueryResponseSource> sources = matches.stream()
                .map(match -> new QueryResponseSource(
                        match.chunkIndex(),
                        match.similarity()))
                .collect(Collectors.toList());

        return new QueryResponse(answer, sources);
    }

    public Flux<String> answerStream(QueryRequest request) {
        Optional<QueryResponse> kgResponse = tryKnowledgeGraph(request);

        if (kgResponse.isPresent()) {
            return Flux.just(kgResponse.get().answer());
        }

        float[] queryEmbedding = embeddingService.embed(request.query());

        List<ChunkSearchResult> matches =
                vectorStoreService.searchSimilar(
                        queryEmbedding,
                        request.resolvedTopK(),
                        request.category());

        if (matches.isEmpty()) {
            return Flux.just(
                    "I could not find relevant information in the knowledge base.");
        }

        String prompt = buildPrompt(request.query(), matches);

        return streamChatCompletion(prompt);
    }

    private Optional<QueryResponse> tryKnowledgeGraph(QueryRequest request) {
        Optional<KgRoutedAnswer> routed =
                questionRouterService.routeToKnowledgeGraph(request.query());

        if (routed.isEmpty()) {
            return Optional.empty();
        }

        KgCountAnswer answer = routed.get().answer();
        String prompt = buildKgPrompt(answer, routed.get().scopeLabel());
        String llmAnswer = invokeChatCompletion(prompt);

        return Optional.of(new QueryResponse(llmAnswer, List.of()));
    }

    private String buildPrompt(
            String question,
            List<ChunkSearchResult> matches) {

        StringBuilder builder = new StringBuilder(
                "Use ONLY the following retrieved information to answer "
                        + "the user's question. If the information is insufficient, "
                        + "say so instead of inventing facts.\n\n");

        matches.forEach(match -> builder.append("[chunk ")
                .append(match.chunkIndex())
                .append("] ")
                .append(match.content())
                .append("\n\n"));

        builder.append("Question: ")
                .append(question)
                .append("\n")
                .append("Answer in a concise paragraph and cite the "
                        + "supporting chunk index in brackets, e.g., [chunk 0].");

        return builder.toString();
    }

    private String buildKgPrompt(
            KgCountAnswer result,
            String scopeLabel) {

        return "You are given a structured fact from a knowledge graph.\n"
                + "Character: " + result.characterName() + "\n"
                + "Lost battles " + scopeLabel + ": " + result.count() + "\n"
                + "Explain this to the user in natural language and note "
                + "that it is based on structured battle data, not long text scanning.";
    }

    private String invokeChatCompletion(String prompt) {
        OllamaChatRequest request = new OllamaChatRequest(
                chatModel,
                List.of(
                        new ChatMessage(
                                "system",
                                "You are a meticulous analyst focused on grounded answers."),
                        new ChatMessage("user", prompt)),
                false,
                new ChatOptions(0.2));

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

                throw new EmbeddingService.OpenAiException(
                        HttpStatus.INTERNAL_SERVER_ERROR,
                        "Ollama returned an empty completion");
            }

            return response.message().content();

        } catch (WebClientResponseException e) {
            log.error(
                    "Ollama chat completion failed: {}",
                    e.getResponseBodyAsString());

            throw new EmbeddingService.OpenAiException(
                    (HttpStatus) e.getStatusCode(),
                    e.getResponseBodyAsString());
        }
    }

    private Flux<String> streamChatCompletion(String prompt) {
        OllamaChatRequest request = new OllamaChatRequest(
                chatModel,
                List.of(
                        new ChatMessage(
                                "system",
                                "You are a meticulous analyst focused on grounded answers."),
                        new ChatMessage("user", prompt)),
                true,
                new ChatOptions(0.2));

        return ollamaWebClient.post()
                .uri("/api/chat")
                .contentType(MediaType.APPLICATION_JSON)
                .accept(MediaType.APPLICATION_NDJSON)
                .bodyValue(request)
                .retrieve()
                .bodyToFlux(String.class)
                .flatMap(this::parseStreamPayload)
                .onErrorResume(WebClientResponseException.class, e -> {
                    log.error(
                            "Ollama streaming failed: {}",
                            e.getResponseBodyAsString());

                    return Flux.error(
                            new EmbeddingService.OpenAiException(
                                    (HttpStatus) e.getStatusCode(),
                                    e.getResponseBodyAsString()));
                });
    }

    private Flux<String> parseStreamPayload(String raw) {
        if (raw == null || raw.isBlank()) {
            return Flux.empty();
        }

        try {
            OllamaChatChunk chunk =
                    objectMapper.readValue(raw.trim(), OllamaChatChunk.class);

            if (chunk.message() == null
                    || chunk.message().content() == null
                    || chunk.message().content().isEmpty()) {
                return Flux.empty();
            }

            return Flux.just(chunk.message().content());

        } catch (Exception e) {
            log.debug("Could not parse Ollama streaming payload: {}", raw);
            return Flux.empty();
        }
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

    private record OllamaChatChunk(OllamaMessage message, boolean done) {
    }

    private record OllamaMessage(String role, String content) {
    }
}
