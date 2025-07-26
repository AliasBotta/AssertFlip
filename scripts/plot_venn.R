# Install and load required packages
if (!require(venn)) {
  install.packages("venn")
  library(venn)
}


# Read and clean the text files
amazon_q <- trimws(readLines(file.path("./data/amazon_q.txt")))
assertflip <- trimws(readLines(file.path("./data/assertflip.txt")))
oterpp <- trimws(readLines(file.path("./data/otterpp.txt")))
openhands <- trimws(readLines(file.path("./data/openhands.txt"))) 

# Filter out empty lines or NAs
amazon_q <- amazon_q[amazon_q != "" & !is.na(amazon_q)]
assertflip <- assertflip[assertflip != "" & !is.na(assertflip)]
oterpp <- oterpp[oterpp != "" & !is.na(oterpp)]
openhands <- openhands[openhands != "" & !is.na(openhands)]

# Create list of sets
sets <- list(
  "Amazon Q" = amazon_q,
  "AssertFlip" = assertflip,
  "Otter++" = oterpp,
  "OpenHands" = openhands
)

# Print counts
cat("Amazon Q:", length(amazon_q), "items\n")
cat("AssertFlip:", length(passinvert), "items\n")
cat("Otter++:", length(oterpp), "items\n")
cat("OpenHands:", length(openhands), "items\n")

# Extract intersection counts
counts_info <- extractInfo(sets, what = "counts")
cat("Intersection counts:\n")
print(counts_info)

# Create PDF with minimal white space
pdf("venn_diagram_4sets.pdf", 
    width = 6, height = 6,           # Square and compact
    paper = "special",
    bg = "white",
    pointsize = 10)

# Set very tight margins
par(mar = c(0.1, 0.1, 0.1, 0.1),    # Almost no margins
    oma = c(0, 0, 0, 0),
    xpd = TRUE)

venn(
  sets,
  ellipse = TRUE,
  ilabels = "counts",
  snames = c("Amazon Q", "AssertFlip", "Otter++", "OpenHands"),
  zcolor = c("#ADD8E6", "#00008B", "#ADD8E6", "#ADD8E6"),
  col = "gray30",
  lwd = 0.5,
  lty = "solid",
  opacity = 0.3,
  cexsn = 3.0, 
  cexcn = 2.0,
  cexil = 1.8, 
  box = FALSE
)

dev.off()
cat("Saved Venn diagram as: venn_diagram_4sets.pdf\n")
