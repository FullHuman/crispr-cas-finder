use crate::cas_types::{
    Cluster, DetectedSystem, GeneDefinition, GeneStatus, HmmerHit, ModelRegistry, SequenceIndex,
    SystemHit, SystemModel,
};
use crate::casparser::GeneCoordinates;
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// In-memory CAS model definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub name: String,
    pub family: String,
    /// XML content of the model definition.
    pub content: String,
}

/// A translated gene record used as an in-memory input to the Cas pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneRecord {
    pub id: String,
    pub protein: String,
    pub start: usize,
    pub end: usize,
    pub strand: String,
}

impl GeneRecord {
    pub fn coordinates(&self) -> GeneCoordinates {
        GeneCoordinates {
            start: self.start,
            end: self.end,
            strand: self.strand.clone(),
        }
    }
}

pub fn gene_coordinates_from_records(records: &[GeneRecord]) -> HashMap<String, GeneCoordinates> {
    records
        .iter()
        .map(|record| (record.id.clone(), record.coordinates()))
        .collect()
}

pub fn build_faa_content(genes: &[GeneRecord]) -> Result<String, String> {
    let mut fasta_amino_acid_content = String::new();
    for gene in genes {
        if !gene.protein.is_ascii() {
            return Err(format!(
                "protein sequence for gene '{}' contains non-ASCII residues",
                gene.id
            ));
        }
        let strand_numeric_value: i32 = if gene.strand == "-" { -1 } else { 1 };
        fasta_amino_acid_content.push_str(&format!(
            ">{id} # {start} # {end} # {strand_int} # ID={id}\n",
            id = gene.id,
            start = gene.start,
            end = gene.end,
            strand_int = strand_numeric_value,
        ));
        for chunk in gene.protein.as_bytes().chunks(60) {
            // The sequence was validated as ASCII above, so every byte boundary is
            // also a UTF-8 boundary.
            fasta_amino_acid_content.push_str(
                std::str::from_utf8(chunk).expect("ASCII protein chunks are valid UTF-8"),
            );
            fasta_amino_acid_content.push('\n');
        }
    }
    Ok(fasta_amino_acid_content)
}

/// Build a model registry from in-memory XML definitions.
pub fn build_model_registry(models: &[ModelDefinition]) -> Result<ModelRegistry, String> {
    let mut registry = ModelRegistry::new();
    for model_def in models {
        let model = parse_cas_model_xml(&model_def.content, &model_def.name, &model_def.family)?;
        registry.add(model);
    }
    Ok(registry)
}

/// Parse a CAS model definition from XML content.
///
/// CRISPRCasFinder definitions use `<system>` roots and non-XML comment lines
/// beginning with `#`. `<model>` roots are accepted for compatibility.
pub fn parse_cas_model_xml(
    content: &str,
    model_name: &str,
    family: &str,
) -> Result<SystemModel, String> {
    let cleaned = strip_hash_comment_lines(content);
    let mut reader = Reader::from_str(&cleaned);
    reader.config_mut().trim_text(true);

    let mut root_attributes = None;
    let mut root_name = None;
    let mut element_stack: Vec<String> = Vec::new();
    let mut genes: Vec<GeneDefinition> = Vec::new();
    let mut current_gene: Option<GeneDefinition> = None;

    loop {
        match reader.read_event() {
            Ok(Event::Start(start)) => {
                let element_name = xml_element_name(&start)?;
                if element_stack.is_empty() {
                    if root_name.is_some() {
                        return Err("CAS model XML contains multiple root elements".to_string());
                    }
                    validate_root_name(&element_name)?;
                    root_attributes = Some(xml_attributes(&start, &reader)?);
                    root_name = Some(element_name.clone());
                } else if element_name == "gene" {
                    if element_stack.len() == 1 {
                        if current_gene.is_some() {
                            return Err("encountered a nested model gene".to_string());
                        }
                        current_gene = Some(parse_gene_definition(&start, &reader)?);
                    } else if element_stack.last().is_some_and(|name| name == "homologs") {
                        add_exchangeable(&mut current_gene, &start, &reader)?;
                    } else {
                        return Err("a nested <gene> must be inside <homologs>".to_string());
                    }
                }
                element_stack.push(element_name);
            }
            Ok(Event::Empty(empty)) => {
                let element_name = xml_element_name(&empty)?;
                if element_stack.is_empty() {
                    if root_name.is_some() {
                        return Err("CAS model XML contains multiple root elements".to_string());
                    }
                    validate_root_name(&element_name)?;
                    root_attributes = Some(xml_attributes(&empty, &reader)?);
                    root_name = Some(element_name);
                } else if element_name == "gene" {
                    if element_stack.len() == 1 {
                        genes.push(parse_gene_definition(&empty, &reader)?);
                    } else if element_stack.last().is_some_and(|name| name == "homologs") {
                        add_exchangeable(&mut current_gene, &empty, &reader)?;
                    } else {
                        return Err("a nested <gene> must be inside <homologs>".to_string());
                    }
                }
            }
            Ok(Event::End(end)) => {
                let ended_name = std::str::from_utf8(end.name().as_ref())
                    .map_err(|error| format!("invalid XML element name: {error}"))?
                    .to_string();
                let opened_name = element_stack
                    .pop()
                    .ok_or_else(|| format!("unexpected closing tag </{ended_name}>"))?;
                if opened_name != ended_name {
                    return Err(format!(
                        "mismatched XML tags: expected </{opened_name}>, found </{ended_name}>"
                    ));
                }
                if ended_name == "gene" && element_stack.len() == 1 {
                    genes.push(
                        current_gene
                            .take()
                            .ok_or_else(|| "model gene ended without a start tag".to_string())?,
                    );
                }
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(error) => return Err(format!("invalid CAS model XML: {error}")),
        }
    }

    if !element_stack.is_empty() {
        return Err("CAS model XML ended before all elements were closed".to_string());
    }
    let root_name = root_name.ok_or_else(|| "CAS model XML has no root element".to_string())?;
    let attributes =
        root_attributes.ok_or_else(|| format!("CAS model <{root_name}> has no attributes"))?;
    let inter_gene_max_space = parse_u32_attribute(&attributes, "inter_gene_max_space", 20)?;
    let min_mandatory_genes_required =
        parse_u32_attribute(&attributes, "min_mandatory_genes_required", 0)?;
    let min_genes_required = parse_u32_attribute(&attributes, "min_genes_required", 0)?;
    let multi_loci = parse_bool_attribute(&attributes, "multi_loci", false)?;

    Ok(SystemModel {
        fully_qualified_name: format!("{family}/{model_name}"),
        family: family.to_string(),
        name: model_name.to_string(),
        version: attributes.get("version").cloned(),
        genes,
        inter_gene_max_space,
        min_mandatory_genes_required,
        min_genes_required,
        multi_loci,
    })
}

pub fn assign_hits_to_model(
    model: &SystemModel,
    hmmer_hits: &HashMap<String, Vec<HmmerHit>>,
    seq_index: &SequenceIndex,
) -> Vec<SystemHit> {
    let mut best_by_protein: HashMap<&str, (SystemHit, bool)> = HashMap::new();

    for gene in &model.genes {
        let gene_names_to_check: Vec<(&str, bool)> = std::iter::once((gene.name.as_str(), false))
            .chain(gene.exchangeables.iter().map(|e| (e.as_str(), true)))
            .collect();

        for (gene_name, is_exchangeable) in gene_names_to_check {
            if let Some(hits) = hmmer_hits.get(gene_name) {
                for hit in hits {
                    if let Some(position) = seq_index.position(&hit.id) {
                        let candidate = SystemHit {
                            hit: hit.clone(),
                            position,
                            gene_ref: gene.name.clone(),
                            gene_status: gene.status,
                            model_fully_qualified_name: model.fully_qualified_name.clone(),
                            is_exchangeable,
                            locus_num: 1,
                            counterpart: String::new(),
                            used_in: Vec::new(),
                        };
                        let inherited = gene.system_ref.is_some();

                        match best_by_protein.get_mut(hit.id.as_str()) {
                            Some((best_hit, best_inherited)) => {
                                if should_replace_assignment(
                                    &candidate,
                                    inherited,
                                    best_hit,
                                    *best_inherited,
                                ) {
                                    *best_hit = candidate;
                                    *best_inherited = inherited;
                                }
                            }
                            None => {
                                best_by_protein.insert(hit.id.as_str(), (candidate, inherited));
                            }
                        }
                    }
                }
            }
        }
    }

    let mut assigned_hits: Vec<SystemHit> =
        best_by_protein.into_values().map(|(hit, _)| hit).collect();
    assigned_hits.sort_by(|left, right| {
        left.position
            .cmp(&right.position)
            .then_with(|| left.gene_ref.cmp(&right.gene_ref))
            .then_with(|| left.hit.id.cmp(&right.hit.id))
    });
    assigned_hits
}

pub fn evaluate_cluster(cluster: &Cluster, model: &SystemModel) -> Option<DetectedSystem> {
    let mandatory_names: HashSet<&str> = model.mandatory_genes().map(|g| g.name.as_str()).collect();
    let accessory_names: HashSet<&str> = model.accessory_genes().map(|g| g.name.as_str()).collect();
    let forbidden_names: HashSet<&str> = model.forbidden_genes().map(|g| g.name.as_str()).collect();

    let found_genes: HashSet<&str> = cluster.hits.iter().map(|h| h.gene_ref.as_str()).collect();

    let mut mandatory_found: Vec<String> = mandatory_names
        .iter()
        .filter(|g| found_genes.contains(*g))
        .map(|g| g.to_string())
        .collect();
    let mut accessory_found: Vec<String> = accessory_names
        .iter()
        .filter(|g| found_genes.contains(*g))
        .map(|g| g.to_string())
        .collect();
    let mut forbidden_found: Vec<String> = forbidden_names
        .iter()
        .filter(|g| found_genes.contains(*g))
        .map(|g| g.to_string())
        .collect();
    mandatory_found.sort_unstable();
    accessory_found.sort_unstable();
    forbidden_found.sort_unstable();

    if !forbidden_found.is_empty() {
        return None;
    }

    let min_mandatory = model.min_mandatory_genes_required as usize;
    let min_genes = model.min_genes_required as usize;
    let total_found = mandatory_found.len() + accessory_found.len();

    if mandatory_found.len() < min_mandatory || total_found < min_genes {
        return None;
    }

    let score: f64 = cluster.hits.iter().map(|h| h.hit.score).sum();
    if !score.is_finite() || cluster.hits.iter().any(|hit| !hit.hit.score.is_finite()) {
        log::warn!(
            "discarding cluster for model {} because it contains a non-finite HMM score",
            model.fully_qualified_name
        );
        return None;
    }
    let max_nb_genes = model.mandatory_count() + model.accessory_count();
    let wholeness = if max_nb_genes > 0 {
        total_found as f64 / max_nb_genes as f64
    } else {
        1.0
    };

    Some(DetectedSystem {
        id: String::new(),
        replicon: String::new(),
        model_fully_qualified_name: model.fully_qualified_name.clone(),
        score,
        wholeness,
        loci_count: 1,
        occurrence: 0,
        hits: cluster.hits.clone(),
        state: "single_locus".to_string(),
        mandatory_found,
        accessory_found,
        forbidden_found,
    })
}

pub fn reverse_complement_dna(seq: &[u8]) -> Vec<u8> {
    crate::dna::reverse_complement(seq)
}

pub fn translate_dna(seq: &[u8], table: &HashMap<[u8; 3], u8>) -> String {
    let mut protein = String::with_capacity(seq.len() / 3);
    for codon in seq.chunks(3) {
        if codon.len() < 3 {
            break;
        }
        let key = [
            codon[0].to_ascii_uppercase(),
            codon[1].to_ascii_uppercase(),
            codon[2].to_ascii_uppercase(),
        ];
        let aa = table.get(&key).copied().unwrap_or(b'X');
        if aa == b'*' {
            break;
        }
        protein.push(aa as char);
    }
    protein
}

pub fn codon_table(_genetic_code: usize) -> HashMap<[u8; 3], u8> {
    const BASES: [u8; 4] = *b"TCAG";
    // Standard code and bacterial code 11 have identical amino-acid mappings.
    // Their difference is alternative initiation codons, which Orphos handles
    // while predicting ORFs rather than while translating internal codons.
    const AMINO_ACIDS: &[u8; 64] =
        b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";

    let mut map = HashMap::with_capacity(64);
    for (index, amino_acid) in AMINO_ACIDS.iter().copied().enumerate() {
        map.insert(
            [BASES[index / 16], BASES[(index / 4) % 4], BASES[index % 4]],
            amino_acid,
        );
    }
    map
}

fn strip_hash_comment_lines(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    for line in content.lines() {
        if line.trim_start().starts_with('#') {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn xml_element_name(element: &BytesStart<'_>) -> Result<String, String> {
    std::str::from_utf8(element.name().as_ref())
        .map(str::to_owned)
        .map_err(|error| format!("invalid XML element name: {error}"))
}

fn xml_attributes(
    element: &BytesStart<'_>,
    reader: &Reader<&[u8]>,
) -> Result<HashMap<String, String>, String> {
    let mut attributes = HashMap::new();
    for attribute in element.attributes() {
        let attribute = attribute.map_err(|error| format!("invalid XML attribute: {error}"))?;
        let name = std::str::from_utf8(attribute.key.as_ref())
            .map_err(|error| format!("invalid XML attribute name: {error}"))?
            .to_string();
        let value = attribute
            .decoded_and_normalized_value(quick_xml::XmlVersion::Implicit1_0, reader.decoder())
            .map_err(|error| format!("invalid value for XML attribute '{name}': {error}"))?
            .into_owned();
        attributes.insert(name, value);
    }
    Ok(attributes)
}

fn validate_root_name(name: &str) -> Result<(), String> {
    if matches!(name, "system" | "model") {
        Ok(())
    } else {
        Err(format!(
            "CAS model root must be <system> or <model>, found <{name}>"
        ))
    }
}

fn parse_gene_definition(
    element: &BytesStart<'_>,
    reader: &Reader<&[u8]>,
) -> Result<GeneDefinition, String> {
    let attributes = xml_attributes(element, reader)?;
    let name = required_attribute(&attributes, "name", "gene")?.to_string();
    let status = match required_attribute(&attributes, "presence", &format!("gene '{name}'"))? {
        "mandatory" => GeneStatus::Mandatory,
        "accessory" => GeneStatus::Accessory,
        "forbidden" => GeneStatus::Forbidden,
        "neutral" => GeneStatus::Neutral,
        presence => {
            return Err(format!(
                "gene '{name}' has unsupported presence value '{presence}'"
            ));
        }
    };
    // Parse this marker even though the concrete alternatives are represented
    // by nested <homologs> elements, so malformed boolean values are reported.
    let _exchangeable = parse_bool_attribute(&attributes, "exchangeable", false)?;

    Ok(GeneDefinition {
        name,
        status,
        loner: parse_bool_attribute(&attributes, "loner", false)?,
        multi_system: parse_bool_attribute(&attributes, "multi_system", false)?,
        system_ref: attributes.get("system_ref").cloned(),
        exchangeables: Vec::new(),
        inter_gene_max_space: parse_optional_u32_attribute(&attributes, "inter_gene_max_space")?,
        multi_model: parse_bool_attribute(&attributes, "multi_model", false)?,
    })
}

fn add_exchangeable(
    current_gene: &mut Option<GeneDefinition>,
    element: &BytesStart<'_>,
    reader: &Reader<&[u8]>,
) -> Result<(), String> {
    let attributes = xml_attributes(element, reader)?;
    let exchangeable_name = required_attribute(&attributes, "name", "homolog gene")?;
    let gene = current_gene
        .as_mut()
        .ok_or_else(|| "encountered <homologs> outside a model gene".to_string())?;
    if !gene
        .exchangeables
        .iter()
        .any(|name| name == exchangeable_name)
    {
        gene.exchangeables.push(exchangeable_name.to_string());
    }
    Ok(())
}

fn required_attribute<'a>(
    attributes: &'a HashMap<String, String>,
    name: &str,
    context: &str,
) -> Result<&'a str, String> {
    attributes
        .get(name)
        .map(String::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{context} is missing required '{name}' attribute"))
}

fn parse_u32_attribute(
    attributes: &HashMap<String, String>,
    name: &str,
    default: u32,
) -> Result<u32, String> {
    parse_optional_u32_attribute(attributes, name).map(|value| value.unwrap_or(default))
}

fn parse_optional_u32_attribute(
    attributes: &HashMap<String, String>,
    name: &str,
) -> Result<Option<u32>, String> {
    attributes
        .get(name)
        .map(|value| {
            value
                .parse::<u32>()
                .map_err(|error| format!("invalid integer for XML attribute '{name}': {error}"))
        })
        .transpose()
}

fn parse_bool_attribute(
    attributes: &HashMap<String, String>,
    name: &str,
    default: bool,
) -> Result<bool, String> {
    match attributes.get(name).map(String::as_str) {
        None => Ok(default),
        Some("1" | "true" | "True") => Ok(true),
        Some("0" | "false" | "False") => Ok(false),
        Some(value) => Err(format!(
            "invalid boolean for XML attribute '{name}': '{value}'"
        )),
    }
}

fn should_replace_assignment(
    candidate: &SystemHit,
    candidate_inherited: bool,
    current: &SystemHit,
    current_inherited: bool,
) -> bool {
    if candidate_inherited != current_inherited {
        return !candidate_inherited;
    }

    candidate
        .hit
        .score
        .total_cmp(&current.hit.score)
        .then_with(|| {
            candidate
                .hit
                .profile_coverage
                .total_cmp(&current.hit.profile_coverage)
        })
        .then_with(|| {
            candidate
                .hit
                .seq_coverage
                .total_cmp(&current.hit.seq_coverage)
        })
        .then_with(|| current.is_exchangeable.cmp(&candidate.is_exchangeable))
        .is_gt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model(genes: Vec<GeneDefinition>) -> SystemModel {
        SystemModel {
            fully_qualified_name: "family/model".to_string(),
            family: "family".to_string(),
            name: "model".to_string(),
            version: None,
            genes,
            inter_gene_max_space: 5,
            min_mandatory_genes_required: 1,
            min_genes_required: 1,
            multi_loci: false,
        }
    }

    fn test_gene(name: &str, status: GeneStatus) -> GeneDefinition {
        GeneDefinition {
            name: name.to_string(),
            status,
            loner: false,
            multi_system: false,
            system_ref: None,
            exchangeables: Vec::new(),
            inter_gene_max_space: None,
            multi_model: false,
        }
    }

    fn test_system_hit(id: &str, gene_ref: &str, position: usize, score: f64) -> SystemHit {
        SystemHit {
            hit: HmmerHit {
                id: id.to_string(),
                gene_name: gene_ref.to_string(),
                seq_len: 100,
                i_evalue: 1e-10,
                score,
                profile_coverage: 0.9,
                seq_coverage: 0.8,
                begin_match: 1,
                end_match: 90,
            },
            position,
            gene_ref: gene_ref.to_string(),
            gene_status: GeneStatus::Mandatory,
            model_fully_qualified_name: "family/model".to_string(),
            is_exchangeable: false,
            locus_num: 1,
            counterpart: String::new(),
            used_in: Vec::new(),
        }
    }

    #[test]
    fn protein_fasta_wraps_at_sixty_ascii_residues() {
        let genes = [GeneRecord {
            id: "gene-1".to_string(),
            protein: "A".repeat(61),
            start: 1,
            end: 183,
            strand: "+".to_string(),
        }];

        let fasta = build_faa_content(&genes).expect("valid protein FASTA");
        let lines: Vec<&str> = fasta.lines().collect();

        assert_eq!(lines[0], ">gene-1 # 1 # 183 # 1 # ID=gene-1");
        assert_eq!(lines[1].len(), 60);
        assert_eq!(lines[2], "A");
    }

    #[test]
    fn protein_fasta_rejects_non_ascii_residues() {
        let genes = [GeneRecord {
            id: "bad-gene".to_string(),
            protein: "MÉT".to_string(),
            start: 1,
            end: 9,
            strand: "+".to_string(),
        }];

        let error = build_faa_content(&genes).expect_err("non-ASCII protein must fail");
        assert!(error.contains("bad-gene"));
        assert!(error.contains("non-ASCII"));
    }

    #[test]
    fn xml_parser_supports_multiline_single_quotes_comments_and_homologs() {
        let xml = r#"
            # This legacy comment line is intentionally not XML.
            <system
                inter_gene_max_space='7'
                min_mandatory_genes_required='1'
                min_genes_required='2'
                multi_loci='true'
                version='2.1'>
                <!-- XML comments are supported too. -->
                <gene
                    name='CasA'
                    presence='mandatory'
                    loner='1'
                    exchangeable='1'>
                    <homologs>
                        <gene name='CasA_alt1' presence='mandatory'/>
                    </homologs>
                    <homologs>
                        <gene name='CasA_alt2' presence='mandatory'/>
                    </homologs>
                </gene>
                <gene name='AntiCas' presence='forbidden'/>
            </system>
        "#;

        let model = parse_cas_model_xml(xml, "TypeX", "CASFinder").expect("parse model");

        assert_eq!(model.fully_qualified_name, "CASFinder/TypeX");
        assert_eq!(model.version.as_deref(), Some("2.1"));
        assert_eq!(model.inter_gene_max_space, 7);
        assert_eq!(model.min_mandatory_genes_required, 1);
        assert_eq!(model.min_genes_required, 2);
        assert!(model.multi_loci);
        assert_eq!(model.genes.len(), 2);
        assert_eq!(model.genes[0].name, "CasA");
        assert_eq!(model.genes[0].status, GeneStatus::Mandatory);
        assert!(model.genes[0].loner);
        assert_eq!(model.genes[0].exchangeables, ["CasA_alt1", "CasA_alt2"]);
        assert_eq!(model.genes[1].status, GeneStatus::Forbidden);
    }

    #[test]
    fn xml_parser_reports_invalid_attributes_and_structure() {
        let invalid_number = r#"<system inter_gene_max_space="many"><gene name="Cas1" presence="mandatory"/></system>"#;
        let malformed = r#"<system><gene name="Cas1" presence="mandatory"/></gene></system>"#;

        let number_error = parse_cas_model_xml(invalid_number, "bad", "family")
            .expect_err("invalid number must fail");
        let structure_error =
            parse_cas_model_xml(malformed, "bad", "family").expect_err("malformed XML must fail");

        assert!(number_error.contains("inter_gene_max_space"));
        assert!(structure_error.contains("invalid CAS model XML"));
    }

    #[test]
    fn assigned_hits_have_deterministic_genomic_order() {
        let model = test_model(vec![test_gene("CasA", GeneStatus::Mandatory)]);
        let hmmer_hits = HashMap::from([(
            "CasA".to_string(),
            vec![
                test_system_hit("protein-2", "CasA", 0, 30.0).hit,
                test_system_hit("protein-1", "CasA", 0, 25.0).hit,
            ],
        )]);
        let sequence_index = SequenceIndex::from_ids(&["protein-1", "protein-2"]);

        let assigned = assign_hits_to_model(&model, &hmmer_hits, &sequence_index);

        assert_eq!(
            assigned
                .iter()
                .map(|hit| hit.hit.id.as_str())
                .collect::<Vec<_>>(),
            ["protein-1", "protein-2"]
        );
    }

    #[test]
    fn cluster_evaluation_is_deterministic_and_rejects_non_finite_scores() {
        let model = test_model(vec![
            test_gene("CasZ", GeneStatus::Mandatory),
            test_gene("CasA", GeneStatus::Mandatory),
        ]);
        let finite_cluster = Cluster {
            hits: vec![
                test_system_hit("protein-1", "CasZ", 0, 30.0),
                test_system_hit("protein-2", "CasA", 1, 25.0),
            ],
            wraps_origin: false,
        };
        let invalid_cluster = Cluster {
            hits: vec![test_system_hit("protein-1", "CasZ", 0, f64::NAN)],
            wraps_origin: false,
        };

        let detected = evaluate_cluster(&finite_cluster, &model).expect("valid system");

        assert_eq!(detected.mandatory_found, ["CasA", "CasZ"]);
        assert_eq!(detected.score, 55.0);
        assert!(evaluate_cluster(&invalid_cluster, &model).is_none());
    }

    #[test]
    fn standard_and_bacterial_codon_maps_are_complete_and_identical() {
        let standard = codon_table(1);
        let bacterial = codon_table(11);

        assert_eq!(standard.len(), 64);
        assert_eq!(standard, bacterial);
        assert_eq!(standard.get(b"TTT"), Some(&b'F'));
        assert_eq!(standard.get(b"ATG"), Some(&b'M'));
        assert_eq!(standard.get(b"TGA"), Some(&b'*'));
    }
}
