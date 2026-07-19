package com.kneeclinic.adminserver.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.kneeclinic.adminserver.entity.Patient;

public interface PatientRepository extends JpaRepository<Patient, Long> {
}
