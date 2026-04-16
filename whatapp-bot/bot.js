const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const axios  = require('axios');

const client = new Client({
    authStrategy: new LocalAuth()
});

const TRIGGER = "ask rag epita";

client.on('qr', qr => {
    qrcode.generate(qr, { small: true });
    console.log('Scan the QR code with WhatsApp');
});

client.on('ready', () => {
    console.log('WhatsApp bot ready!');
});

client.on('message', async msg => {
    const text   = msg.body.trim();
    const sender = msg.from;

    console.log(`Message from ${sender}: ${text}`);

    // Only respond to messages starting with trigger
    if (!text.toLowerCase().startsWith(TRIGGER)) return;

    // Strip trigger prefix to get the actual question
    const question = text.slice(TRIGGER.length).trim();

    if (!question) {
        await msg.reply("Please include a question after the trigger.\nExample: ASKING RAG EPITA What do I do if someone is choking?");
        return;
    }

    console.log(`[RAG] Question: ${question}`);
    await msg.reply("⏳ Looking up your question...");

    try {
        const response = await axios.post('http://127.0.0.1:5000/query', {
            question: question
        });
        await msg.reply(response.data.answer);
    } catch (err) {
        console.error('RAG error:', err.message);
        await msg.reply('Sorry, something went wrong. Please try again.');
    }
});

client.initialize();