package com.bio.detect;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;

/**
 * Pine Nematode Detection System - Main Application
 * CT图像松材线虫病检测系统
 */
@SpringBootApplication
@EnableAsync
public class PineNematodeApplication {
    
    public static void main(String[] args) {
        SpringApplication.run(PineNematodeApplication.class, args);
        System.out.println("============================================");
        System.out.println("  松材线虫病检测系统已启动！");
        System.out.println("  访问地址: http://localhost:8080/api");
        System.out.println("============================================");
    }
}
