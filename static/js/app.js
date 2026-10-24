const chat = document.getElementById('chat');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');
const newThreadId = () => crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
let threadId = newThreadId();
const welcomeHtml = chat.innerHTML;
const chatList = document.getElementById('chatList');
let chats = JSON.parse(localStorage.getItem('chats') || '[]');

function saveMessage(id, title, message){
  let item = chats.find(c => c.id === id);
  if(!item){ item = {id, title: title.slice(0, 60), messages: []}; chats.unshift(item); }
  item.messages.push(message);
  localStorage.setItem('chats', JSON.stringify(chats));
  renderChatList();
}

function renderChatList(){
  chatList.replaceChildren(...chats.map(c => {
    const item = document.createElement('div');
    item.className = `chat-item${c.id === threadId ? ' active' : ''}`;
    item.title = c.title;
    item.onclick = () => openChat(c.id);
    const title = document.createElement('span');
    title.className = 'chat-title';
    title.textContent = c.title;
    const remove = document.createElement('button');
    remove.className = 'chat-delete';
    remove.textContent = '×';
    remove.title = 'Delete chat';
    remove.onclick = e => { e.stopPropagation(); deleteChat(c.id); };
    item.append(title, remove);
    return item;
  }));
}

function deleteChat(id){
  if(!confirm('Delete this chat?')) return;
  chats = chats.filter(c => c.id !== id);
  localStorage.setItem('chats', JSON.stringify(chats));
  fetch(`/api/chat/${encodeURIComponent(id)}`, {method: 'DELETE'});
  if(id === threadId) newChat(); else renderChatList();
}

function resetView(){
  chat.innerHTML = welcomeHtml;
  trace.innerHTML = '<div class="empty">No request yet.</div>';
  sourceUsed.textContent = '—';
}

function openChat(id){
  threadId = id;
  resetView();
  chats.find(c => c.id === id).messages.forEach(m => addMessage(m.role, m.text, m.source, m.citations));
  renderChatList();
}

function newChat(){
  threadId = newThreadId();
  resetView();
  renderChatList();
  question.focus();
}
document.getElementById('newChat').addEventListener('click', newChat);
renderChatList();

function escapeHtml(s=''){return s.replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function formatText(s=''){return escapeHtml(s).replace(/\n/g,'<br>');}
function bubbleHtml(text, source='', citations=[]){
  const citeHtml=citations.length?`<div class="citations"><strong>Sources</strong><br>${citations.map(c=>c.url?`<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>`:escapeHtml(c.title)).join('<br>')}</div>`:'';
  return `<div class="avatar">AI</div><div class="bubble">${formatText(text)}${source?`<div class="answer-source">Source: ${escapeHtml(source)}</div>`:''}${citeHtml}</div>`;
}
function setMessage(wrap, text, source='', citations=[]){wrap.innerHTML=bubbleHtml(text,source,citations); chat.scrollTop=chat.scrollHeight;}
function addMessage(role, text, source='', citations=[]){
  const wrap=document.createElement('div'); wrap.className=`message ${role}`;
  chat.appendChild(wrap); setMessage(wrap,text,source,citations);
  return wrap;
}
function renderTrace(items=[]){trace.innerHTML=items.length?items.map(x=>`<div class="trace-item">${escapeHtml(x)}</div>`).join(''):'<div class="empty">No trace.</div>';}
async function askAgent(q){
  const id=threadId;
  addMessage('user',q); saveMessage(id,q,{role:'user',text:q}); question.value=''; renderTrace(['Running LangGraph workflow...']); sourceUsed.textContent='Running';
  const btn=form.querySelector('button'); btn.disabled=true;
  const reply=addMessage('assistant','…'); const steps=[]; let text='';
  const handle=ev=>{
    if(ev.type==='token'){text+=ev.data; setMessage(reply,text);}
    else if(ev.type==='trace'){steps.push(ev.data); renderTrace(steps);}
    else if(ev.type==='done'){setMessage(reply,ev.answer,ev.source_used,ev.citations||[]); renderTrace(ev.trace||[]); sourceUsed.textContent=ev.source_used; saveMessage(id,q,{role:'assistant',text:ev.answer,source:ev.source_used,citations:ev.citations||[]});}
    else if(ev.type==='error'){throw new Error(ev.data);}
  };
  try{
    const res=await fetch('/api/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,thread_id:id})});
    if(!res.ok||!res.body) throw new Error('Request failed');
    const reader=res.body.getReader(); const decoder=new TextDecoder(); let buffer='';
    while(true){
      const {value,done}=await reader.read(); if(done) break;
      buffer+=decoder.decode(value,{stream:true});
      const lines=buffer.split('\n'); buffer=lines.pop();
      lines.filter(l=>l.trim()).forEach(l=>handle(JSON.parse(l)));
    }
    if(buffer.trim()) handle(JSON.parse(buffer));
  }catch(e){setMessage(reply,`Error: ${e.message}`); renderTrace(['Request failed']); sourceUsed.textContent='Error';}
  finally{btn.disabled=false;}
}
form.addEventListener('submit',e=>{e.preventDefault(); const q=question.value.trim(); if(q) askAgent(q);});
document.querySelectorAll('.example').forEach(b=>b.addEventListener('click',()=>askAgent(b.textContent.trim())));

const modal=document.getElementById('uploadModal');
document.getElementById('openUpload').onclick=()=>modal.classList.remove('hidden');
document.getElementById('closeUpload').onclick=()=>modal.classList.add('hidden');
document.getElementById('uploadBtn').onclick=async()=>{
  const file=document.getElementById('fileInput').files[0]; const key=document.getElementById('adminKey').value; const status=document.getElementById('uploadStatus');
  if(!file){status.textContent='Choose a file first.'; return;}
  status.textContent='Indexing document...'; const fd=new FormData(); fd.append('file',file);
  try{const r=await fetch('/api/ingest',{method:'POST',headers:{'X-Admin-Key':key},body:fd}); const d=await r.json(); if(!r.ok) throw new Error(d.detail||'Upload failed'); status.textContent=`Indexed ${d.file}: ${d.chunks} chunks.`;}catch(e){status.textContent=`Error: ${e.message}`;}
};
