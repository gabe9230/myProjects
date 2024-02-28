let input = prompt("Input: ").split('')
// let key = prompt("(enter a number please) key: ")
let output
const d = new Date();
// let time = d.getTime() * key;
let charArr = [0,1,2,3,4,5,6,7,8,9,"a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
// console.log(time)
function InputNumerifier(x) {
    for (let i = 0; i < x.length; i++) {
        for (let q = 0; q < charArr.length; q++) {
            if (x[i] === charArr[q]) {
                x[i] = q
            }
        }
    }
    for (let i = 0; i < x.length; i++) {
        x[i] = parseInt(x[i])
    }
    return x
}
function OutputDeNumerifier(x) {
    for (let i = 0; i < x.length; i++) {
        x[i] = charArr[x[i]]
    }
    return x
}
input = (InputNumerifier(input))

function shift(i) {
    for (let i = 0; i < input.length; i++) {

    }
}