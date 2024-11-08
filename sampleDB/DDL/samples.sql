DROP TABLE IF EXISTS samples;

CREATE TABLE samples (
    sample_id SERIAL PRIMARY KEY,
    sample_name VARCHAR(100) NOT NULL,
    sample_description TEXT,
    sample_prep TEXT,
    sample_keywords VARCHAR(255),
    sample_owner VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
grant all on schema public to app;
grant all on samples to app;
grant all on sequence samples_sample_id_seq to app;