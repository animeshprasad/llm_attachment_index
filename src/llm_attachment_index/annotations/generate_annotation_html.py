import json
from pathlib import Path
from html import escape

# Input/output setup — fully portable
json_dir = Path("src/llm_attachment_index/annotations/data_for_annotation")
output_dir = Path("src/llm_attachment_index/annotations/html_for_annotation")
output_dir.mkdir(parents=True, exist_ok=True)

json_files = sorted(json_dir.glob("*.json"))
n_files = len(json_files)

# ---------- index.html ----------
first_file_stem = json_files[0].stem if json_files else "0"
index_html = f"""
<!DOCTYPE html>
<html>
<head>
  <title>Annotation Instructions</title>
  <style>
    body {{ font-family: sans-serif; padding: 2em; max-width: 700px; margin: auto; }}
    input, button {{ font-size: 1rem; padding: 0.5em; }}
  </style>
</head>
<body>
  <h1>Annotation Instructions</h1>
  <p>Please read before annotating:</p>
  <ul>
    <li>Carefully read the full conversation.</li>
    <li>Use the dropdowns to rate each field objectively (0–5).</li>
    <li><strong>Select text spans</strong> by clicking and dragging over important parts of the conversation.</li>
    <li>Selected spans will be highlighted and can be managed in the "Text Spans" section.</li>
    <li><strong>All changes are automatically saved</strong> as you work and when you navigate between files.</li>
    <li>Use the "Download All Annotations" button on any page to export your complete annotations.</li>
    <li>Use navigation buttons to move between files.</li>
  </ul>
  <label>Enter your name: <input type="text" id="annotatorName" /></label>
  <br><br>
  <button onclick="start()">Start</button>

  <script>
    // Load saved name when page loads
    window.onload = function() {{
      const savedName = localStorage.getItem("annotator_name");
      if (savedName) {{
        document.getElementById("annotatorName").value = savedName;
      }}
    }};

    async function start() {{
      const name = document.getElementById("annotatorName").value;
      if (!name.trim()) {{
        alert("Please enter your name.");
        return;
      }}
      localStorage.setItem("annotator_name", name);
      window.location.href = "annot_{first_file_stem}.html";
    }}
  </script>
</body>
</html>
"""
(output_dir / "index.html").write_text(index_html)

# ---------- Template for each annotation page ----------
def create_annotation_page(idx, json_obj, total, file_list):
    conversation = json_obj.get("conversation_history", [])
    
    # Field mapping: field_name -> list of options
    field_options = {
        "Demonstrated Attachment Style": ["None", "Anxious", "Dismissive", "Fearful", "Secure"],
        "Potentially Problematic": ["No", "Yes"]
    }
    
    fields = list(field_options.keys())

    def make_select(name):
        options = field_options.get(name, ["Option 1", "Option 2"])  # fallback options
        return f"""
        <label><strong>{name}:</strong>
          <select id="{name}">
            <option value="">-- Select --</option>
            {''.join(f'<option value="{option}">{option}</option>' for option in options)}
          </select>
        </label><br><br>
        """

    # Generate chat HTML with unique IDs for each turn and character indexing
    chat_html = ""
    for turn_idx, turn in enumerate(conversation):
        role = turn.get("role", "")
        msg = escape(turn.get("content", ""))
        align = "user" if role == "user" else "assistant"
        chat_html += f'<div class="chat {align}" data-turn="{turn_idx}"><div class="bubble" id="turn-{turn_idx}">{msg}</div></div>\n'

    nav = []
    if idx > 0:
        prev_file = file_list[idx-1].stem
        nav.append(f'<a href="annot_{prev_file}.html">Previous</a>')
    nav.append(f'<a href="index.html">Home</a>')
    if idx < total - 1:
        next_file = file_list[idx+1].stem
        nav.append(f'<a href="annot_{next_file}.html">Next</a>')

    selects = ''.join(make_select(f) for f in fields)

    return f"""
<!DOCTYPE html>
<html>
<head>
  <title>Annotation {idx+1}</title>
  <style>
    body {{ font-family: sans-serif; padding: 2em; max-width: 800px; margin: auto; }}
    .chat {{ display: flex; margin: 10px 0; }}
    .chat.user {{ justify-content: flex-end; }}
    .chat.assistant {{ justify-content: flex-start; }}
    .bubble {{
      padding: 10px 15px;
      border-radius: 15px;
      max-width: 70%;
      white-space: pre-wrap;
      user-select: text;
      position: relative;
    }}
    .user .bubble {{
      background-color: #d1e7dd;
      color: #000;
      border-top-right-radius: 0;
    }}
    .assistant .bubble {{
      background-color: #f8d7da;
      color: #000;
      border-top-left-radius: 0;
    }}
    .nav {{ margin-top: 2em; }}
    select {{ font-size: 1rem; padding: 0.2em; }}
    
    /* Highlight styles */
    .highlight {{
      background-color: #ffeb3b;
      border-radius: 3px;
      padding: 1px 2px;
      cursor: pointer;
    }}
    .highlight:hover {{
      background-color: #ffc107;
    }}
    
    /* Span management styles */
    .span-manager {{
      margin-top: 1em;
      padding: 1em;
      border: 1px solid #ddd;
      border-radius: 5px;
      background-color: #f9f9f9;
    }}
    .span-item {{
      margin: 0.5em 0;
      padding: 0.5em;
      border: 1px solid #ccc;
      border-radius: 3px;
      background-color: white;
    }}
    .span-text {{
      font-style: italic;
      margin: 0.25em 0;
    }}
    .span-meta {{
      font-size: 0.8em;
      color: #666;
    }}
    .span-comment {{
      margin-top: 0.5em;
    }}
    .span-comment label {{
      display: block;
      font-weight: bold;
      margin-bottom: 0.25em;
      font-size: 0.9em;
    }}
    .span-comment textarea {{
      width: 100%;
      min-height: 60px;
      padding: 0.5em;
      border: 1px solid #ddd;
      border-radius: 3px;
      font-family: inherit;
      font-size: 0.9em;
      resize: vertical;
    }}
    .span-comment textarea:focus {{
      outline: none;
      border-color: #007bff;
    }}
    .delete-span {{
      background-color: #dc3545;
      color: white;
      border: none;
      padding: 0.25em 0.5em;
      border-radius: 3px;
      cursor: pointer;
      margin-left: 0.5em;
    }}
    .delete-span:hover {{
      background-color: #c82333;
    }}
    
    /* Selection mode indicator */
    .selection-mode {{
      background-color: #e3f2fd;
      padding: 0.5em;
      border-radius: 3px;
      margin-bottom: 1em;
      text-align: center;
    }}
    
    /* Download button styling */
    .download-btn {{
      margin-top: 1em;
      padding: 0.5em 1em;
      background-color: #28a745;
      color: white;
      border: none;
      border-radius: 3px;
      cursor: pointer;
      font-size: 1rem;
    }}
    .download-btn:hover {{
      background-color: #218838;
    }}
  </style>
</head>
<body>
  <h2>Annotator: <span id="annotatorNameDisplay"></span></h2>
  <div class="nav">{' | '.join(nav)}</div>
  <hr>
  
  <div id="conversationContainer">
    {chat_html}
  </div>
  
  <hr>
  
  <div class="span-manager">
    <h3>Text Spans</h3>
    <button onclick="enterSelectionMode()">Add New Span</button>
    
    <div id="selectionMode" class="selection-mode" style="display: none;">
      <strong>Selection Mode Active:</strong> Click and drag to select text spans in the conversation above. Click "Save Selection" when done.
      <button onclick="saveCurrentSelection()">Save Selection</button>
      <button onclick="cancelSelection()">Cancel</button>
    </div>
    
    <div id="spansList"></div>
  </div>
  
  <hr>
  <h3>Annotation</h3>
  <p><em>All changes are automatically saved</em></p>
  {selects}
  <button onclick='exportAnnotations()' class="download-btn">Download All Annotations</button>
  <div class="nav">{' | '.join(nav)}</div>
  <script>
    const fields = {fields};
    const fname = "annot_{file_list[idx].stem}";
    const annotator = localStorage.getItem("annotator_name") || "unknown";
    document.getElementById("annotatorNameDisplay").textContent = annotator;
    
    let selectionMode = false;
    let currentSelection = null;
    let spans = [];

    // Load from localStorage
    const data = JSON.parse(localStorage.getItem(fname) || "{{}}");
    for (let f of fields) {{
      if (data[f]) document.getElementById(f).value = data[f];
    }}
    
    // Load spans
    if (data.spans) {{
      spans = data.spans;
      // Ensure all spans have a comment field (for backward compatibility)
      spans.forEach(span => {{
        if (!span.hasOwnProperty('comment')) {{
          span.comment = "";
        }}
      }});
      restoreHighlights();
      updateSpansList();
    }}
    
    // Auto-save on field changes
    for (let f of fields) {{
      const field = document.getElementById(f);
      field.addEventListener('change', saveToStorage);
    }}
    
    // Auto-save before page unload or navigation
    window.addEventListener('beforeunload', saveToStorage);
    
    // Auto-save when clicking navigation links
    document.querySelectorAll('a[href]').forEach(link => {{
      link.addEventListener('click', function(e) {{
        saveToStorage();
        // Small delay to ensure save completes
        setTimeout(() => {{
          window.location.href = this.href;
        }}, 100);
        e.preventDefault();
      }});
    }});

    function enterSelectionMode() {{
      selectionMode = true;
      document.getElementById("selectionMode").style.display = "block";
      document.addEventListener("mouseup", handleMouseUp);
    }}
    
    function exitSelectionMode() {{
      selectionMode = false;
      document.getElementById("selectionMode").style.display = "none";
      document.removeEventListener("mouseup", handleMouseUp);
      clearSelection();
    }}
    
    function handleMouseUp() {{
      if (!selectionMode) return;
      
      const selection = window.getSelection();
      if (selection.rangeCount === 0 || selection.toString().trim() === "") return;
      
      const range = selection.getRangeAt(0);
      const commonAncestor = range.commonAncestorContainer;
      
      // Find the turn container
      let turnElement = commonAncestor.nodeType === Node.TEXT_NODE ? 
        commonAncestor.parentElement : commonAncestor;
      
      while (turnElement && !turnElement.id?.startsWith("turn-")) {{
        turnElement = turnElement.parentElement;
      }}
      
      if (!turnElement) return;
      
      const turnId = turnElement.id;
      const turnIndex = parseInt(turnId.split("-")[1]);
      
      // Calculate character indices within the turn
      const turnText = turnElement.textContent;
      const selectedText = selection.toString();
      
      // Find start and end positions
      const startOffset = getTextOffset(turnElement, range.startContainer, range.startOffset);
      const endOffset = startOffset + selectedText.length;
      
      currentSelection = {{
        turnIndex: turnIndex,
        startOffset: startOffset,
        endOffset: endOffset,
        text: selectedText,
        turnText: turnText
      }};
    }}
    
    function getTextOffset(container, node, offset) {{
      let textOffset = 0;
      const walker = document.createTreeWalker(
        container,
        NodeFilter.SHOW_TEXT,
        null,
        false
      );
      
      let currentNode;
      while (currentNode = walker.nextNode()) {{
        if (currentNode === node) {{
          return textOffset + offset;
        }}
        textOffset += currentNode.textContent.length;
      }}
      return textOffset;
    }}
    
    function saveCurrentSelection() {{
      if (!currentSelection) {{
        alert("No text selected. Please select some text first.");
        return;
      }}
      
      // Add to spans array
      const spanId = Date.now().toString();
      const span = {{
        id: spanId,
        comment: "", // Initialize with empty comment
        ...currentSelection
      }};
      
      spans.push(span);
      updateSpansList();
      restoreHighlights();
      saveToStorage(); // Auto-save spans
      exitSelectionMode();
    }}
    
    function cancelSelection() {{
      currentSelection = null;
      exitSelectionMode();
    }}
    
    function clearSelection() {{
      if (window.getSelection) {{
        window.getSelection().removeAllRanges();
      }}
    }}
    
    function saveToStorage() {{
      // Save current state to localStorage
      const result = {{}};
      result.annotator = annotator;
      for (let f of fields) {{
        result[f] = document.getElementById(f).value;
      }}
      result.spans = spans;
      localStorage.setItem(fname, JSON.stringify(result));
    }}
    
    function updateSpansList() {{
      const spansList = document.getElementById("spansList");
      if (spans.length === 0) {{
        spansList.innerHTML = "<p>No text spans selected yet.</p>";
        return;
      }}
      
      spansList.innerHTML = spans.map(span => `
        <div class="span-item">
          <div class="span-text">"${{span.text}}"</div>
          <div class="span-meta">Turn ${{span.turnIndex + 1}}, characters ${{span.startOffset}}-${{span.endOffset}}</div>
          <div class="span-comment">
            <label>Comment (why did you select this?):</label>
            <textarea 
              id="comment-${{span.id}}" 
              placeholder="Enter your comment about this text span..."
              onchange="updateSpanComment('${{span.id}}', this.value)"
              oninput="updateSpanComment('${{span.id}}', this.value)"
            >${{span.comment || ""}}</textarea>
          </div>
          <button class="delete-span" onclick="deleteSpan('${{span.id}}')">Delete</button>
        </div>
      `).join("");
    }}
    
    function updateSpanComment(spanId, comment) {{
      const span = spans.find(s => s.id === spanId);
      if (span) {{
        span.comment = comment;
        saveToStorage(); // Auto-save when comment changes
      }}
    }}
    
    function deleteSpan(spanId) {{
      spans = spans.filter(span => span.id !== spanId);
      updateSpansList();
      restoreHighlights();
      saveToStorage(); // Auto-save spans
    }}
    
    function restoreHighlights() {{
      // Clear existing highlights
      document.querySelectorAll(".highlight").forEach(el => {{
        const parent = el.parentNode;
        parent.replaceChild(document.createTextNode(el.textContent), el);
        parent.normalize();
      }});
      
      // Apply highlights for each span
      spans.forEach(span => {{
        const turnElement = document.getElementById(`turn-${{span.turnIndex}}`);
        if (!turnElement) return;
        
        highlightTextInElement(turnElement, span.startOffset, span.endOffset);
      }});
    }}
    
    function highlightTextInElement(element, startOffset, endOffset) {{
      const walker = document.createTreeWalker(
        element,
        NodeFilter.SHOW_TEXT,
        null,
        false
      );
      
      let currentOffset = 0;
      let textNodes = [];
      let node;
      
      // Collect all text nodes and their positions
      while (node = walker.nextNode()) {{
        textNodes.push({{
          node: node,
          start: currentOffset,
          end: currentOffset + node.textContent.length
        }});
        currentOffset += node.textContent.length;
      }}
      
      // Find nodes that intersect with our target range
      textNodes.forEach(textNode => {{
        const nodeStart = Math.max(startOffset - textNode.start, 0);
        const nodeEnd = Math.min(endOffset - textNode.start, textNode.node.textContent.length);
        
        if (nodeStart < nodeEnd && nodeEnd > 0) {{
          const beforeText = textNode.node.textContent.substring(0, nodeStart);
          const highlightText = textNode.node.textContent.substring(nodeStart, nodeEnd);
          const afterText = textNode.node.textContent.substring(nodeEnd);
          
          const parent = textNode.node.parentNode;
          const fragment = document.createDocumentFragment();
          
          if (beforeText) {{
            fragment.appendChild(document.createTextNode(beforeText));
          }}
          
          const highlight = document.createElement("span");
          highlight.className = "highlight";
          highlight.textContent = highlightText;
          fragment.appendChild(highlight);
          
          if (afterText) {{
            fragment.appendChild(document.createTextNode(afterText));
          }}
          
          parent.replaceChild(fragment, textNode.node);
        }}
      }});
    }}

    async function exportAnnotations() {{
      // Ensure current page is saved first (extra safety)
      saveToStorage();
      const name = localStorage.getItem("annotator_name") || "unknown";
      const data = {{}};
      // Get all localStorage keys that start with "annot_"
      for (let i = 0; i < localStorage.length; i++) {{
        const key = localStorage.key(i);
        if (key && key.startsWith("annot_")) {{
          const val = localStorage.getItem(key);
          if (val) data[key] = JSON.parse(val);
        }}
      }}
      const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: "application/json" }});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `annotations_${{name}}.json`;
      a.click();
    }}
  </script>
</body>
</html>
"""

# ---------- Generate all annotation pages ----------
for i, file in enumerate(json_files):
    with open(file) as f:
        data = json.load(f)
    html = create_annotation_page(i, data, n_files, json_files)
    (output_dir / f"annot_{file.stem}.html").write_text(html)
