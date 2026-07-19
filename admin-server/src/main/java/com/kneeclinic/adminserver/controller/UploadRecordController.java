package com.kneeclinic.adminserver.controller;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import com.kneeclinic.adminserver.dto.PredictResponse;
import com.kneeclinic.adminserver.entity.Patient;
import com.kneeclinic.adminserver.entity.UploadRecord;
import com.kneeclinic.adminserver.repository.PatientRepository;
import com.kneeclinic.adminserver.repository.UploadRecordRepository;
import com.kneeclinic.adminserver.service.AiClientService;

import lombok.RequiredArgsConstructor;

@RestController
@RequestMapping("/api/patients/{patientId}/uploads")
@RequiredArgsConstructor
public class UploadRecordController {

    private final PatientRepository patientRepository;
    private final UploadRecordRepository uploadRecordRepository;
    private final AiClientService aiClientService;

    @PostMapping
    public UploadRecord upload(@PathVariable Long patientId, @RequestParam("file") MultipartFile file) {
        Patient patient = patientRepository.findById(patientId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "patient not found: " + patientId));

        PredictResponse result = aiClientService.predict(file);

        UploadRecord record = new UploadRecord();
        record.setPatient(patient);
        record.setImageFilename(file.getOriginalFilename());
        record.setPredictedGrade(result.getPredictedGrade());
        record.setConfidence(result.getConfidence());
        record.setReport(result.getReport());
        return uploadRecordRepository.save(record);
    }

    @GetMapping
    public List<UploadRecord> list(@PathVariable Long patientId) {
        return uploadRecordRepository.findByPatientIdOrderByCreatedAtDesc(patientId);
    }
}
