package com.kneeclinic.adminserver.controller;

import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

import com.kneeclinic.adminserver.entity.Patient;
import com.kneeclinic.adminserver.repository.PatientRepository;

import lombok.RequiredArgsConstructor;

@RestController
@RequestMapping("/api/patients")
@RequiredArgsConstructor
public class PatientController {

    private final PatientRepository patientRepository;

    @GetMapping
    public List<Patient> list() {
        return patientRepository.findAll();
    }

    @GetMapping("/{id}")
    public Patient get(@PathVariable Long id) {
        return patientRepository.findById(id)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "patient not found: " + id));
    }

    @PostMapping
    public Patient create(@RequestBody Patient patient) {
        patient.setId(null);
        return patientRepository.save(patient);
    }
}
