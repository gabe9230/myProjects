class HaasGCodeGenerator {
    constructor(programNumber = '0020_', programName = 'Project Number __') {
      this.gCode = [];
      this.programNumber = programNumber;
      this.programName = programName;
      this.currentTool = 1;
      this.currentSpindleSpeed = 2000;
      this.currentFeedRate = 10;
      this.safeZ = 2.0;
      this.rapidPlane = 0.1;
      this.toolOffset = 1;
      this.workOffset = 'G54';
      this.decimalPlaces = 4;
    }
  
    // Core formatting methods
    addLine(line = '') {
      this.gCode.push(line);
    }
  
    comment(text) {
      this.addLine(`(${text})`);
    }
  
    // Program structure
    programStart() {
      this.addLine('%');
      this.addLine(`O${this.programNumber}_ (${this.programName})`);
      this.addLine('G17 G40 G80 G54 G20');
      this.addLine('G28 G91 G0 X0 Y0 Z0');
    }
  
    toolChange(toolNumber, toolDescription = '') {
      this.currentTool = toolNumber;
      this.addLine(`M06 T${toolNumber} (Tool: ${toolDescription})`);
    }
  
    startPosition(x, y, spindleSpeed = null) {
      if (spindleSpeed !== null) this.currentSpindleSpeed = spindleSpeed;
      this.addLine(`G0 G90 ${this.workOffset} X${x.toFixed(this.decimalPlaces)} Y${y.toFixed(this.decimalPlaces)} S${this.currentSpindleSpeed} M3`);
      this.addLine(`G43 H${this.toolOffset} Z${this.rapidPlane.toFixed(this.decimalPlaces)} M8`);
    }
  
    plungeIntoPart(z, feedRate = null) {
      if (feedRate !== null) this.currentFeedRate = feedRate;
      this.addLine(`G1 Z${z.toFixed(this.decimalPlaces)} F${this.currentFeedRate}`);
    }
  
    programEnd() {
      this.addLine('G40 G80');
      this.addLine(`G0 Z${this.safeZ.toFixed(this.decimalPlaces)} M9`);
      this.addLine('G28 G91 G0 X0 Y0 Z0 M5');
      this.addLine('M30');
      this.addLine('%');
    }
  
    // Movement commands
    rapidMove(x, y, z) {
      this.addLine(`G0 X${x.toFixed(this.decimalPlaces)} Y${y.toFixed(this.decimalPlaces)} Z${z.toFixed(this.decimalPlaces)}`);
    }
  
    linearMove(x, y, z, feedRate = null) {
      if (feedRate !== null) this.currentFeedRate = feedRate;
      this.addLine(`G1 X${x.toFixed(this.decimalPlaces)} Y${y.toFixed(this.decimalPlaces)} Z${z.toFixed(this.decimalPlaces)} F${this.currentFeedRate}`);
    }
  
    // Machining operations
    drillHole(x, y, startZ, depth, peckDepth = 0, dwell = 0) {
      this.comment(`Drilling hole at X${x} Y${y}`);
      this.rapidMove(x, y, this.rapidPlane);
      this.plungeIntoPart(startZ);
      
      if (peckDepth > 0) {
        // G83 Peck drilling cycle
        this.addLine(`G83 X${x.toFixed(this.decimalPlaces)} Y${y.toFixed(this.decimalPlaces)} Z${depth.toFixed(this.decimalPlaces)} Q${peckDepth.toFixed(this.decimalPlaces)} R${startZ.toFixed(this.decimalPlaces)} P${dwell.toFixed(2)} F${this.currentFeedRate}`);
      } else {
        // G81 Simple drilling cycle
        this.addLine(`G81 X${x.toFixed(this.decimalPlaces)} Y${y.toFixed(this.decimalPlaces)} Z${depth.toFixed(this.decimalPlaces)} R${startZ.toFixed(this.decimalPlaces)} F${this.currentFeedRate}${dwell > 0 ? ` P${dwell.toFixed(2)}` : ''}`);
      }
      
      this.addLine('G80'); // Cancel cycle
      this.rapidMove(x, y, this.rapidPlane);
    }
  
    faceMill(startX, startY, width, length, depth, stepDown, toolDiameter) {
      this.comment(`Facing operation: ${width}x${length} area to depth ${depth}`);
      const radius = toolDiameter / 2;
      const passes = Math.ceil(Math.abs(depth) / Math.abs(stepDown));
      
      for (let pass = 1; pass <= passes; pass++) {
        const currentDepth = Math.min(pass * stepDown, depth);
        this.comment(`Pass ${pass} at depth ${currentDepth.toFixed(this.decimalPlaces)}`);
        
        // Face milling pattern (left to right, stepping over)
        for (let y = startY + radius; y <= startY + length - radius; y += toolDiameter * 0.8) {
          this.rapidMove(startX + radius, y, this.rapidPlane);
          this.linearMove(startX + radius, y, currentDepth);
          this.linearMove(startX + width - radius, y, currentDepth);
          this.rapidMove(startX + width - radius, y, this.rapidPlane);
        }
      }
      
      this.rapidMove(startX + radius, startY + radius, this.rapidPlane);
    }
  
    // Generate the complete program
    getGCode() {
      return this.gCode.join('\n');
    }
  }
  
  // Example usage following your template exactly
  function generateProgram() {
    const haas = new HaasGCodeGenerator('0020', 'Sample Part');
    
    // Program setup
    haas.programStart();
    haas.toolChange(1, '1/2" END MILL');
    haas.startPosition(1.0, 1.0, 2500);
    haas.plungeIntoPart(-0.1, 10);
    
    // Machining operations (defined with JS variables)
    const partWidth = 5.0;
    const partLength = 3.0;
    const toolDiameter = 0.5;
    
    // Face milling operation
    haas.faceMill(0, 0, partWidth, partLength, -0.1, 0.05, toolDiameter);
    
    // Drill holes operation
    const holes = [
      {x: 0.5, y: 0.5},
      {x: partWidth-0.5, y: 0.5},
      {x: partWidth-0.5, y: partLength-0.5},
      {x: 0.5, y: partLength-0.5}
    ];
    
    haas.comment('Drilling holes');
    haas.plungeIntoPart(0.1); // Return to safe plane
    holes.forEach(hole => {
      haas.drillHole(hole.x, hole.y, 0.1, -0.6, 0.1, 0.5);
    });
    
    // Program end
    haas.programEnd();
    
    return haas.getGCode();
  }
  
  // Convert JS to G-code
  function convertToHaasGCode(jsFunction) {
    try {
      return jsFunction();
    } catch (e) {
      return `(Error generating G-code: ${e.message})`;
    }
  }
  
  // Generate the sample program
  const haasGCode = convertToHaasGCode(generateProgram);
  console.log(haasGCode);