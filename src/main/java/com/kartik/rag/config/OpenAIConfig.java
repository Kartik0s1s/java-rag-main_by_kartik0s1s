
package com.kartik.rag.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class OpenAIConfig {

    @Bean
    public WebClient.Builder webClientBuilder() {
        return WebClient.builder()
                .exchangeStrategies(
                        ExchangeStrategies.builder()
                                .codecs(codecs -> codecs.defaultCodecs()
                                        .maxInMemorySize(16 * 1024 * 1024))
                                .build());
    }

    @Bean
    public WebClient ollamaWebClient(
            WebClient.Builder webClientBuilder,
            @Value("${ollama.base-url:http://localhost:11434}")
            String baseUrl) {

        return webClientBuilder
                .baseUrl(baseUrl)
                .build();
    }
}
