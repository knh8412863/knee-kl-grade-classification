package com.kneeclinic.adminserver.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import com.kneeclinic.adminserver.entity.UploadRecord;

public interface UploadRecordRepository extends JpaRepository<UploadRecord, Long> {

    List<UploadRecord> findByPatientIdOrderByCreatedAtDesc(Long patientId);
}
