package com.kneeclinic.adminserver.entity;

import java.time.LocalDateTime;

import org.hibernate.annotations.CreationTimestamp;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Lob;
import jakarta.persistence.ManyToOne;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/** One knee X-ray upload and its AI prediction result, tied to a patient. */
@Entity
@Getter
@Setter
@NoArgsConstructor
public class UploadRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "patient_id", nullable = false)
    private Patient patient;

    private String imageFilename;

    private Integer predictedGrade;

    private Double confidence;

    @Lob
    @Column(columnDefinition = "TEXT")
    private String report;

    @CreationTimestamp
    private LocalDateTime createdAt;
}
