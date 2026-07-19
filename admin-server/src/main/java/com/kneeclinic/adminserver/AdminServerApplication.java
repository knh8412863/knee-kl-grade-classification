package com.kneeclinic.adminserver;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class AdminServerApplication {

	public static void main(String[] args) {
		loadDotenvIntoSystemProperties();
		SpringApplication.run(AdminServerApplication.class, args);
	}

	/**
	 * Loads KEY=VALUE pairs from a ".env" file (shared with the Python side, at the
	 * repo root) into JVM system properties, so secrets like the DB password don't
	 * need to be duplicated or hardcoded in application.properties. Checks the
	 * current working directory first, then one level up, since this app may be
	 * launched from either the repo root or the admin-server module directory.
	 */
	private static void loadDotenvIntoSystemProperties() {
		for (String candidate : new String[] { ".env", "../.env" }) {
			Path path = Path.of(candidate);
			if (!Files.isRegularFile(path)) {
				continue;
			}
			try {
				for (String line : Files.readAllLines(path)) {
					String trimmed = line.trim();
					if (trimmed.isEmpty() || trimmed.startsWith("#")) {
						continue;
					}
					int eq = trimmed.indexOf('=');
					if (eq <= 0) {
						continue;
					}
					String key = trimmed.substring(0, eq).trim();
					String value = trimmed.substring(eq + 1).trim();
					if (value.length() >= 2 && value.startsWith("\"") && value.endsWith("\"")) {
						value = value.substring(1, value.length() - 1);
					}
					if (System.getProperty(key) == null) {
						System.setProperty(key, value);
					}
				}
			} catch (IOException e) {
				throw new IllegalStateException("Failed to read .env file at " + path, e);
			}
			return;
		}
	}

}
