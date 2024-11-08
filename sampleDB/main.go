package main

import (
	"context"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
)

// Sample represents a sample record in the database
type Sample struct {
	ID          int
	Name        string
	Description string
	Keywords    string
	Owner       string
}

// Initialize a global database connection pool
var dbPool *pgxpool.Pool

func main() {
	// Connect to PostgreSQL
	var err error
	dbURL := "postgres://app:app@localhost:5432/sampledb" // replace with your credentials
	dbPool, err = pgxpool.New(context.Background(), dbURL)
	if err != nil {
		log.Fatalf("Unable to connect to database: %v\n", err)
	}
	defer dbPool.Close()

	// Start server

	http.HandleFunc("/", mainPageHandler)
	http.HandleFunc("/samples/edit/", editSampleHandler)
	http.HandleFunc("/samples/", sampleDetailHandler) // Handle viewing sample details
	http.HandleFunc("/samples/new", newSampleHandler)

	fmt.Println("Server started at :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

// mainPageHandler serves the main page and handles search functionality
func mainPageHandler(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("query")
	var samples []Sample
	var err error

	if query != "" {
		// If there's a search query, fetch matching samples
		samples, err = searchSamples(query)
		if err != nil {
			http.Error(w, "Error retrieving search results", http.StatusInternalServerError)
			return
		}
	} else {
		// Otherwise, fetch all samples
		samples, err = getAllSamples()
		if err != nil {
			http.Error(w, "Error retrieving samples", http.StatusInternalServerError)
			return
		}
	}

	tmpl, err := template.ParseFiles("templates/main.html")
	if err != nil {
		http.Error(w, "Error loading template", http.StatusInternalServerError)
		return
	}

	tmpl.Execute(w, samples)
}

// searchSamples queries the database for samples by name or keywords
func searchSamples(query string) ([]Sample, error) {
	// Split the query into individual keywords
	keywords := strings.FieldsFunc(query, func(r rune) bool {
		return r == ' ' || r == ',' || r == ';'
	})

	// Construct the SQL query with `ANY` and `string_to_array`
	var whereClauses []string
	var args []interface{}
	for i, keyword := range keywords {
		whereClauses = append(whereClauses, fmt.Sprintf("$%d = ANY(string_to_array(sample_keywords, ','))", i+1))
		args = append(args, keyword)
	}
	whereClause := strings.Join(whereClauses, " OR ")

	// Also match the full query string in `sample_name`
	queryText := fmt.Sprintf(`
        SELECT sample_id, sample_name, sample_description, sample_keywords, sample_owner
        FROM samples
        WHERE sample_name ILIKE $%d OR %s`, len(args)+1, whereClause)
	args = append(args, "%"+query+"%")

	// Execute the query
	rows, err := dbPool.Query(context.Background(), queryText, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	// Collect the results
	var samples []Sample
	for rows.Next() {
		var sample Sample
		err := rows.Scan(&sample.ID, &sample.Name, &sample.Description, &sample.Keywords, &sample.Owner)
		if err != nil {
			return nil, err
		}
		samples = append(samples, sample)
	}

	return samples, nil
}

// getAllSamples retrieves all samples when there’s no search query
func getAllSamples() ([]Sample, error) {
	rows, err := dbPool.Query(context.Background(),
		"SELECT sample_id, sample_name, sample_description, sample_keywords, sample_owner FROM samples")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var samples []Sample
	for rows.Next() {
		var sample Sample
		err := rows.Scan(&sample.ID, &sample.Name, &sample.Description, &sample.Keywords, &sample.Owner)
		if err != nil {
			return nil, err
		}
		samples = append(samples, sample)
	}

	return samples, nil
}

// getSamples retrieves all samples from the database
func getSamples() ([]Sample, error) {
	rows, err := dbPool.Query(context.Background(), "SELECT sample_id, sample_name, sample_description, sample_keywords, sample_owner FROM samples")
	if err != nil {
		fmt.Printf("%s\n", err)
		return nil, err
	}
	defer rows.Close()

	var samples []Sample
	for rows.Next() {
		// fmt.Printf("parsing rows\n")
		var s Sample
		err := rows.Scan(&s.ID, &s.Name, &s.Description, &s.Keywords, &s.Owner)
		if err != nil {
			return nil, err
		}
		samples = append(samples, s)
	}

	return samples, nil
}
func sampleDetailHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract sample ID from URL by removing "/samples/"
	sampleID := strings.TrimPrefix(r.URL.Path, "/samples/")

	// Retrieve sample details from the database
	sample, err := getSampleByID(sampleID)
	if err != nil {
		http.Error(w, "Sample not found", http.StatusNotFound)
		return
	}

	// Load the sample detail template
	tmpl, err := template.ParseFiles("templates/sample_detail.html")
	if err != nil {
		http.Error(w, "Error loading template", http.StatusInternalServerError)
		return
	}

	tmpl.Execute(w, sample)
}

// getSampleByID retrieves a sample from the database by its ID
func getSampleByID(sampleID string) (Sample, error) {
	var sample Sample
	err := dbPool.QueryRow(context.Background(),
		"SELECT sample_id, sample_name, sample_description, sample_keywords, sample_owner FROM samples WHERE sample_id=$1", sampleID).Scan(
		&sample.ID, &sample.Name, &sample.Description, &sample.Keywords, &sample.Owner)
	return sample, err
}

// editSampleHandler updates a sample's details in the database
func editSampleHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract sample ID from URL by removing "/samples/edit/"
	sampleID := strings.TrimPrefix(r.URL.Path, "/samples/edit/")

	// Parse form data
	err := r.ParseForm()
	if err != nil {
		http.Error(w, "Error parsing form", http.StatusBadRequest)
		return
	}

	name := r.FormValue("name")
	description := r.FormValue("description")
	keywords := r.FormValue("keywords")
	owner := r.FormValue("owner")

	// Update sample in the database
	_, err = dbPool.Exec(context.Background(),
		"UPDATE samples SET sample_name=$1, sample_description=$2, sample_keywords=$3, sample_owner=$4 WHERE sample_id=$5",
		name, description, keywords, owner, sampleID)
	if err != nil {
		http.Error(w, "Error updating sample", http.StatusInternalServerError)
		return
	}

	// Redirect to the sample's detail page to display updated info
	http.Redirect(w, r, "/samples/"+sampleID, http.StatusSeeOther)
}

// newSampleFormHandler displays the form to add a new sample
func newSampleFormHandler(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles("templates/new_sample.html")
	if err != nil {
		http.Error(w, "Error loading template", http.StatusInternalServerError)
		return
	}
	tmpl.Execute(w, nil)
}

// createSampleHandler processes the new sample form submission
func createSampleHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse form data
	name := r.FormValue("name")
	description := r.FormValue("description")
	keywords := r.FormValue("keywords")
	owner := r.FormValue("owner")

	// Insert the new sample into the database
	_, err := dbPool.Exec(context.Background(),
		"INSERT INTO samples (sample_name, sample_description, sample_keywords, sample_owner) VALUES ($1, $2, $3, $4)",
		name, description, keywords, owner)
	if err != nil {
		http.Error(w, "Error adding sample", http.StatusInternalServerError)
		return
	}

	// Redirect to the main page to show the new sample
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// newSampleHandler handles both displaying the form and processing the submission
func newSampleHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		// Display the form
		tmpl, err := template.ParseFiles("templates/new_sample.html")
		if err != nil {
			http.Error(w, "Error loading template", http.StatusInternalServerError)
			return
		}
		tmpl.Execute(w, nil)
	} else if r.Method == http.MethodPost {
		// Process the form submission
		name := r.FormValue("name")
		description := r.FormValue("description")
		keywords := r.FormValue("keywords")
		owner := r.FormValue("owner")

		// Insert the new sample into the database
		_, err := dbPool.Exec(context.Background(),
			"INSERT INTO samples (sample_name, sample_description, sample_keywords, sample_owner) VALUES ($1, $2, $3, $4)",
			name, description, keywords, owner)
		if err != nil {
			fmt.Printf("%v", err)
			http.Error(w, "Error adding sample", http.StatusInternalServerError)
			return
		}

		// Redirect to the main page to show the new sample
		http.Redirect(w, r, "/", http.StatusSeeOther)
	} else {
		// Return 405 Method Not Allowed for other methods
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}
