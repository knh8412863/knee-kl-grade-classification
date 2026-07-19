package com.kneeclinic.adminserver.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Getter;
import lombok.Setter;

/** Mirrors the JSON body returned by the Python FastAPI /predict endpoint. */
@Getter
@Setter
public class PredictResponse {

    @JsonProperty("predicted_grade")
    private Integer predictedGrade;

    @JsonProperty("grade_description")
    private String gradeDescription;

    private Double confidence;

    @JsonProperty("grade_probabilities")
    private List<Double> gradeProbabilities;

    @JsonProperty("gradcam_image_base64")
    private String gradcamImageBase64;

    private String report;
}
