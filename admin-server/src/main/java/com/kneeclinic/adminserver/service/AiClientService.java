package com.kneeclinic.adminserver.service;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

import com.kneeclinic.adminserver.dto.PredictResponse;

/**
 * Calls the Python FastAPI AI server's /predict endpoint.
 *
 * <p>The multipart/form-data body is built by hand (rather than via
 * {@code MultipartBodyBuilder}) because that builder targets Spring's reactive
 * multipart writer, which doesn't play well with RestClient's blocking request
 * factories here and silently produces a body FastAPI can't parse.
 */
@Service
public class AiClientService {

    private static final String CRLF = "\r\n";

    private final RestClient restClient;

    public AiClientService(@Value("${ai.server.base-url:http://localhost:8000}") String baseUrl) {
        // The JDK HttpClient-based default request factory attempts an HTTP/2 (h2c)
        // upgrade that uvicorn (HTTP/1.1-only) can't handle, corrupting the request.
        // SimpleClientHttpRequestFactory is plain HTTP/1.1 and avoids that entirely.
        this.restClient = RestClient.builder()
                .baseUrl(baseUrl)
                .requestFactory(new SimpleClientHttpRequestFactory())
                .build();
    }

    public PredictResponse predict(MultipartFile file) {
        String boundary = "AdminServerBoundary" + UUID.randomUUID();
        byte[] body = buildMultipartBody(boundary, file);

        return restClient.post()
                .uri("/predict")
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .body(body)
                .retrieve()
                .body(PredictResponse.class);
    }

    private byte[] buildMultipartBody(String boundary, MultipartFile file) {
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            String contentType = file.getContentType() != null ? file.getContentType() : "application/octet-stream";

            out.write(("--" + boundary + CRLF).getBytes(StandardCharsets.UTF_8));
            out.write(("Content-Disposition: form-data; name=\"file\"; filename=\""
                    + file.getOriginalFilename() + "\"" + CRLF).getBytes(StandardCharsets.UTF_8));
            out.write(("Content-Type: " + contentType + CRLF + CRLF).getBytes(StandardCharsets.UTF_8));
            out.write(file.getBytes());
            out.write((CRLF + "--" + boundary + "--" + CRLF).getBytes(StandardCharsets.UTF_8));

            return out.toByteArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
