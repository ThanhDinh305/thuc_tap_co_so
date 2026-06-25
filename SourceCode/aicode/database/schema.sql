-- ============================================================
--  Fruit Recognition Web App - MySQL Schema
--  Database: fruit_recognition_db
-- ============================================================

CREATE DATABASE IF NOT EXISTS fruit_recognition_db
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE fruit_recognition_db;

-- ─────────────────────────────────────────
--  TABLE: users
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
  id         INT          AUTO_INCREMENT PRIMARY KEY,
  name       VARCHAR(100) NOT NULL,
  email      VARCHAR(150) NOT NULL UNIQUE,
  password   VARCHAR(255) NOT NULL,
  role       ENUM('user', 'admin') NOT NULL DEFAULT 'user',
  avatar     VARCHAR(500) NULL,
  created_at TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_users_email (email),
  INDEX idx_users_role  (role)
) ENGINE=InnoDB;

-- ─────────────────────────────────────────
--  TABLE: predictions
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
  id            INT            AUTO_INCREMENT PRIMARY KEY,
  user_id       INT            NOT NULL,
  image_path    VARCHAR(500)   NOT NULL,
  fruit_name    VARCHAR(100)   NOT NULL,
  confidence    DECIMAL(5,2)   NOT NULL COMMENT 'Percentage 0-100',
  ripeness      VARCHAR(50)    NULL,
  calories      DECIMAL(8,2)   NULL,
  carbohydrates DECIMAL(8,2)   NULL,
  protein       DECIMAL(8,2)   NULL,
  fat           DECIMAL(8,2)   NULL,
  fiber         DECIMAL(8,2)   NULL,
  sugar         DECIMAL(8,2)   NULL,
  vitamins      JSON           NULL COMMENT 'Full nutrition JSON',
  created_at    TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX idx_pred_user_id    (user_id),
  INDEX idx_pred_fruit_name (fruit_name),
  INDEX idx_pred_created_at (created_at)
) ENGINE=InnoDB;

-- ─────────────────────────────────────────
--  SEED: Default admin account
--  Email: admin@admin.com
--  Password: admin123 (bcrypt hashed)
-- ─────────────────────────────────────────
INSERT IGNORE INTO users (name, email, password, role)
VALUES (
  'Administrator',
  'admin@admin.com',
  '$2b$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi',
  'admin'
);
-- NOTE: The hash above is bcrypt of "admin123"
-- You can regenerate with: node -e "const b=require('bcryptjs');b.hash('admin123',10).then(console.log)"
